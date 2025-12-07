from multiprocessing import Process
import time
import torch
from torch.nn import functional as F
import torch.nn as nn

from replay_buffer import ReplayBuffer
from model import CNNModel

class Learner(Process):
    def __init__(self, config, replay_buffer, shared_model_data):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.shared_model_data = shared_model_data
        self.current_version = -1

    def run(self):
        device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

        model = CNNModel().to(device)

        use_dataparallel = False
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            use_dataparallel = True
            print(f"[Learner] Using DataParallel on {torch.cuda.device_count()} GPUs")

        # 初始化 shared_model_data
        state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        self.shared_model_data['state_dict'] = cpu_state
        self.current_version += 1
        self.shared_model_data['version'] = self.current_version

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])

        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)

        cur_time = time.time()
        iterations = 0
        while True:
            batch = self.replay_buffer.sample(self.config['batch_size'])

            obs = torch.tensor(batch['state']['observation'], dtype=torch.float32).to(device)
            mask = torch.tensor(batch['state']['action_mask'], dtype=torch.float32).to(device)
            actions = torch.tensor(batch['action'], dtype=torch.long).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv'], dtype=torch.float32).to(device)
            targets = torch.tensor(batch['target'], dtype=torch.float32).to(device)

            states = {
                'observation': obs,
                'action_mask': mask
            }

            print('Iteration %d, replay buffer in %d out %d' % (
                iterations,
                self.replay_buffer.stats.get('sample_in', 0),
                self.replay_buffer.stats.get('sample_out', 0)
            ))

            model.train()
            with torch.no_grad():
                old_logits, _ = model(states)
                # mask logits 防止 NaN
                old_logits = old_logits.masked_fill(mask == 0, -1e9)
                old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
                old_log_probs = torch.log(old_probs + 1e-8).detach()

            for _ in range(self.config['epochs']):
                logits, values = model(states)
                # mask logits 防止 NaN
                logits = logits.masked_fill(mask == 0, -1e9)

                probs = F.softmax(logits, dim=1).gather(1, actions)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)

                surr1 = ratio * advs.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs.unsqueeze(-1)
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                action_dist = torch.distributions.Categorical(logits=logits)
                entropy_loss = -torch.mean(action_dist.entropy())

                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防梯度爆炸
                optimizer.step()

            # 更新 shared_model_data
            state_dict = model.module.state_dict() if use_dataparallel else model.state_dict()
            cpu_state = {k: v.cpu() for k, v in state_dict.items()}
            self.current_version += 1
            self.shared_model_data['state_dict'] = cpu_state
            self.shared_model_data['version'] = self.current_version

            # 保存 checkpoint
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                torch.save(cpu_state, path)
                cur_time = t

            iterations += 1
