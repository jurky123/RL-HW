import os
import time
import traceback
from multiprocessing import Process
import sys
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from replay_buffer import ReplayBuffer
from model import CNNModel

LEARNER_LOG_FILE = 'learner_crash.log'


class Learner(Process):
    def __init__(self, config, replay_buffer, shared_model_data):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.shared_model_data = shared_model_data
        self.current_version = -1

    def _save_cpu_state(self, model_state, iterations):
        path_dir = self.config['ckpt_save_path']
        os.makedirs(path_dir, exist_ok=True)
        path = os.path.join(path_dir, f"model_new.pt")
        torch.save(model_state, path)
        print(f"[Learner] Saved checkpoint at iteration {iterations}: {path}")

    def run(self):
        try:
            # 确保日志目录存在
            log_dir = self.config['ckpt_save_path']
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, LEARNER_LOG_FILE)
            
            # 打开日志文件，将 stdout 和 stderr 重定向到该文件
            self._log_file = open(log_path, 'a', buffering=1) # 立即刷新 (buffering=1)
            sys.stdout = self._log_file
            sys.stderr = self._log_file
            
            # 可选：使用 logging 模块
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Learner PID %(process)d] - %(levelname)s - %(message)s', stream=sys.stdout)
            logger = logging.getLogger('Learner')
            
            print("-" * 50)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Learner] Process starting, output redirected to {log_path}")
            print("-" * 50)
            
        except Exception as setup_e:
            # 如果日志设置失败，则打印到原始 stderr 并退出
            print(f"FATAL: Learner failed to setup logging: {setup_e}", file=sys.stderr)
            return # Setup failed, exit process
        try:
            device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
            print(f"[Learner] PID {os.getpid()} starting on device {device}")

            model = CNNModel().to(device)
            use_dataparallel = False
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                use_dataparallel = True
                print(f"[Learner] Using DataParallel on {torch.cuda.device_count()} GPUs")

            # publish initial model
            state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
            self.shared_model_data['state_dict'] = {k: v.cpu() for k, v in state_dict.items()}
            self.current_version += 1
            self.shared_model_data['version'] = self.current_version

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=1e-4)

            # wait for min samples
            print("[Learner] Waiting for replay buffer to have min samples:", self.config['min_sample'])
            wait_t = 0
            while self.replay_buffer.size() < self.config['min_sample']:
                time.sleep(0.1)
                if wait_t % 20 == 0: print(f"[Learner] Current replay buffer size: {self.replay_buffer.size()}")
                wait_t += 1
            print("[Learner] Minimum samples reached, starting training.")
            cur_time = time.time()
            iterations = 0

            while True:
                batch = self.replay_buffer.sample(self.config['batch_size'])



                obs = torch.tensor(batch['state']['observation'], dtype=torch.float32, device=device)
                mask = torch.tensor(batch['state']['action_mask'], dtype=torch.float32, device=device)
                actions = torch.tensor(batch['action'], dtype=torch.long, device=device).unsqueeze(-1)
                advs = torch.tensor(batch['adv'], dtype=torch.float32, device=device)
                targets = torch.tensor(batch['target'], dtype=torch.float32, device=device)

                # 修复 NaN/Inf
                obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
                advs = torch.nan_to_num(advs, nan=0.0, posinf=1e6, neginf=-1e6)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

                # normalize advantages
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                advs = torch.clamp(advs, -20, 20)

                states = {'observation': obs, 'action_mask': mask}

                print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))

                model.train()

                # compute old logits safely
                with torch.no_grad():
                    old_logits, _ = model(states)
                    old_logits = torch.nan_to_num(old_logits, nan=0.0, posinf=1e6, neginf=-1e6)

                    # ensure each row has at least one valid action
                    valid_action_counts = (mask > 0.5).sum(dim=1)
                    invalid_rows = (valid_action_counts == 0).nonzero().squeeze(-1)
                    for i in invalid_rows:
                        mask[i, 0] = 1.0

                    masked_old_logits = old_logits.masked_fill(mask == 0, -1e9)
                    row_max = masked_old_logits.max(dim=1).values
                    masked_old_logits[row_max == float('-inf'), 0] = 0.0

                    old_probs = F.softmax(masked_old_logits, dim=1).gather(1, actions)
                    old_log_probs = torch.log(old_probs + 1e-8).detach()

                # PPO epochs
                for _ in range(self.config['epochs']):
                    logits, values = model(states)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                    values = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

                    masked_logits = logits.masked_fill(mask == 0, -1e9)
                    row_max = masked_logits.max(dim=1).values
                    masked_logits[row_max == float('-inf'), 0] = 0.0

                    probs = F.softmax(masked_logits, dim=1).gather(1, actions)
                    log_probs = torch.log(probs + 1e-8)
                    ratio = torch.exp(log_probs - old_log_probs)
                    ratio = torch.clamp(ratio, 0.0, 10.0)

                    advs_exp = advs.unsqueeze(-1)
                    surr1 = ratio * advs_exp
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs_exp
                    policy_loss = -torch.mean(torch.min(surr1, surr2))

                    value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))

                    action_dist = torch.distributions.Categorical(logits=masked_logits)
                    entropy_loss = -torch.mean(action_dist.entropy())

                    loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.get('max_grad_norm', 0.5))
                    optimizer.step()

                # publish model
                state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
                cpu_state = {k: v.cpu() for k, v in state_dict.items()}
                self.current_version += 1
                self.shared_model_data['state_dict'] = cpu_state
                self.shared_model_data['version'] = self.current_version

                # checkpoint
                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    try:
                        self._save_cpu_state(cpu_state, iterations)
                    except Exception as e:
                        print("[Learner] Failed to save checkpoint:", e)
                        traceback.print_exc()
                    cur_time = t

                iterations += 1

        except KeyboardInterrupt:
            print("[Learner] KeyboardInterrupt, exiting.")
        except Exception as e:
            print("[Learner] Exception in run():", e)
            traceback.print_exc()
        finally:
            # 无论如何，确保关闭日志文件
            if hasattr(self, '_log_file') and self._log_file:
                print("[Learner] Process finished, closing log file.")
                self._log_file.close()