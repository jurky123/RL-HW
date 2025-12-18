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
        # 初始化版本号，如果 Launcher 加载了历史模型，则同步该版本号
        self.current_version = shared_model_data.get('version', -1)

    def _save_cpu_state(self, model_state, iterations, optimizer_state=None):
        path_dir = self.config['ckpt_save_path']
        os.makedirs(path_dir, exist_ok=True)
        
        # 1. 保存用于下次自动续传的文件 (路径与 Launcher 中的 latest_model_path 一致)
        path_latest = os.path.join(path_dir, "latest_model.pt")
        
        save_data = {
            'version': self.current_version,
            'state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'iterations': iterations
        }
        
        torch.save(save_data, path_latest)
        
        # 2. 同时保留一个带编号的备份，方便回溯历史性能
        path_backup = os.path.join(path_dir, f"model_iter_{iterations}.pt")
        torch.save(model_state, path_backup) 
        
        print(f"[Learner] Saved checkpoint at iteration {iterations}: {path_latest}")

    def run(self):
        # --- 日志重定向设置 ---
        try:
            log_dir = self.config['ckpt_save_path']
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, LEARNER_LOG_FILE)
            
            self._log_file = open(log_path, 'a', buffering=1) 
            sys.stdout = self._log_file
            sys.stderr = self._log_file
            
            logging.basicConfig(level=logging.INFO, 
                                format='%(asctime)s - [Learner PID %(process)d] - %(levelname)s - %(message)s', 
                                stream=sys.stdout)
            
            print("-" * 50)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Learner] Process starting...")
            print("-" * 50)
        except Exception as setup_e:
            print(f"FATAL: Learner failed to setup logging: {setup_e}", file=sys.stderr)
            return 

        try:
            device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
            
            # --- 核心修改：模型初始化与历史权重加载 ---
            model = CNNModel().to(device)
            
            # 如果 shared_model_data 中已有数据，说明 Launcher 成功加载了历史模型
            if self.shared_model_data.get('state_dict') is not None:
                print(f"[Learner] Detected historical model. Loading state_dict (Version: {self.current_version})...")
                model.load_state_dict(self.shared_model_data['state_dict'])
            else:
                print("[Learner] No historical model found. Starting from scratch.")

            use_dataparallel = False
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                use_dataparallel = True
                print(f"[Learner] Using DataParallel on {torch.cuda.device_count()} GPUs")

            # --- 优化器初始化 ---
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=1e-4)

            # --- 等待数据采样 ---
            print(f"[Learner] Waiting for replay buffer to reach min samples: {self.config['min_sample']}")
            while self.replay_buffer.size() < self.config['min_sample']:
                time.sleep(1.0)
                print(f"[Learner] Current buffer size: {self.replay_buffer.size()}")

            print("[Learner] Minimum samples reached, starting training.")
            cur_time = time.time()
            iterations = 0

            # --- 训练主循环 ---
            while True:
                batch = self.replay_buffer.sample(self.config['batch_size'])

                # 数据上屏与预处理
                obs = torch.tensor(batch['state']['observation'], dtype=torch.float32, device=device)
                mask = torch.tensor(batch['state']['action_mask'], dtype=torch.float32, device=device)
                actions = torch.tensor(batch['action'], dtype=torch.long, device=device).unsqueeze(-1)
                advs = torch.tensor(batch['adv'], dtype=torch.float32, device=device)
                targets = torch.tensor(batch['target'], dtype=torch.float32, device=device)

                # 异常值处理
                obs = torch.nan_to_num(obs, nan=0.0)
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                advs = torch.clamp(advs, -10.0, 10.0)

                states = {'observation': obs, 'action_mask': mask}
                model.train()

                # 计算旧策略概率 (PPO 核心)
                with torch.no_grad():
                    old_logits, _ = model(states)
                    masked_old_logits = old_logits.masked_fill(mask == 0, -1e9)
                    old_probs = F.softmax(masked_old_logits, dim=1).gather(1, actions)
                    old_log_probs = torch.log(old_probs + 1e-8).detach()

                # PPO 多次迭代更新
                for _ in range(self.config['epochs']):
                    logits, values = model(states)
                    masked_logits = logits.masked_fill(mask == 0, -1e9)
                    
                    probs = F.softmax(masked_logits, dim=1).gather(1, actions)
                    log_probs = torch.log(probs + 1e-8)
                    
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advs.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs.unsqueeze(-1)
                    
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    value_loss = F.mse_loss(values.squeeze(-1), targets)
                    
                    action_dist = torch.distributions.Categorical(logits=masked_logits)
                    entropy_loss = -torch.mean(action_dist.entropy())

                    loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.get('max_grad_norm', 0.5))
                    optimizer.step()

                # --- 1. 同步权重到共享内存 (Actor 才能拉取到更新) ---
                state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
                cpu_state = {k: v.cpu() for k, v in state_dict.items()}
                
                self.current_version += 1
                self.shared_model_data['state_dict'] = cpu_state
                self.shared_model_data['version'] = self.current_version

                # --- 2. 定期保存到硬盘 ---
                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    try:
                        self._save_cpu_state(cpu_state, iterations, optimizer.state_dict())
                        cur_time = t
                    except Exception as e:
                        print(f"[Learner] Save failed: {e}")

                if iterations % 10 == 0:
                    print(f"[Iter {iterations}] Loss: {loss.item():.4f} | PLoss: {policy_loss.item():.4f} | VLoss: {value_loss.item():.4f}")

                iterations += 1

        except KeyboardInterrupt:
            print("[Learner] KeyboardInterrupt exit.")
        except Exception as e:
            print(f"[Learner] Runtime Error: {e}")
            traceback.print_exc()
        finally:
            if hasattr(self, '_log_file'):
                self._log_file.close()