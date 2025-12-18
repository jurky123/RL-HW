from multiprocessing import Process
import numpy as np
import torch
import time

from replay_buffer import ReplayBuffer
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel
from model_expert import ResnetModel

class Actor(Process):
    def __init__(self, config, replay_buffer, shared_model_data):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.shared_model_data = shared_model_data

    def run(self):
        torch.set_num_threads(1)

        # create network model (on CPU)
        model = CNNModel()

        # wait for learner to publish initial model
        print(f"[{self.name}] Waiting for initial model from Learner...")
        while self.shared_model_data['version'] == -1:
            time.sleep(0.1)

        latest_version_id = self.shared_model_data['version']
        state_dict = self.shared_model_data['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        version_id = latest_version_id
        print(f"[{self.name}] Loaded initial model version {latest_version_id}")

        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})

        for episode in range(self.config['episodes_per_actor']):
            # check for updated model
            latest_id = self.shared_model_data['version']
            if latest_id > version_id:
                state_dict = self.shared_model_data['state_dict']
                model.load_state_dict(state_dict)
                model.eval()
                version_id = latest_id
                print(f"[{self.name}] Updated model to version {version_id}")

            obs = env.reset()
            # prepare storage
            episode_data = {
                agent_name: {
                    'state': {'observation': [], 'action_mask': []},
                    'action': [], 'reward': [], 'value': []
                } for agent_name in env.agent_names
            }

            done = False
            while not done:
                actions = {}
                values = {}

                # for each player in the current observation dict
                for agent_name, state in obs.items():
                    agent_data = episode_data[agent_name]

                    # store raw obs/mask (numpy)
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])

                    # convert to tensors (CPU) with correct dtype
                    state_obs = torch.tensor(state['observation'], dtype=torch.float32).unsqueeze(0)
                    state_mask = torch.tensor(state['action_mask'], dtype=torch.float32).unsqueeze(0)
                    state_tensor = {'observation': state_obs, 'action_mask': state_mask}

                    # model inference
                    model.eval()
                    with torch.no_grad():
                        logits, value = model(state_tensor)
                        # mask logits here for sampling (do not change model forward)
                        masked_logits = logits.masked_fill(state_mask == 0, -1e9)
                        action_dist = torch.distributions.Categorical(logits=masked_logits)
                        action = int(action_dist.sample().item())
                        value = float(value.item())

                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(action)
                    agent_data['value'].append(value)

                # interact with env
                next_obs, rewards, done = env.step(actions)
                # append rewards for players that exist in rewards
                for agent_name, r in rewards.items():
                    episode_data[agent_name]['reward'].append(r)
                obs = next_obs

            # log
            print(self.name, 'Episode', episode, 'Model', version_id, 'Reward', rewards)

            # postprocess per agent
            for agent_name, agent_data in episode_data.items():
                # skip agents with no collected steps
                if len(agent_data['state']['observation']) == 0:
                    # debug log to trace why empty
                    print(f"[{self.name}] Skipping agent {agent_name} — no steps collected in this episode.")
                    continue

                # ensure rewards/actions alignment
                if len(agent_data['action']) < len(agent_data['reward']):
                    # if reward list longer by 1 due to env terminal step alignment
                    agent_data['reward'].pop(0)

                # stack arrays
                try:
                    obs_np = np.stack(agent_data['state']['observation']).astype(np.float32)
                    mask_np = np.stack(agent_data['state']['action_mask']).astype(np.float32)
                except Exception as e:
                    print(f"[{self.name}] Error stacking data for {agent_name}: {e}")
                    continue

                actions = np.array(agent_data['action'], dtype=np.int64)
                rewards = np.array(agent_data['reward'], dtype=np.float32)
                values = np.array(agent_data['value'], dtype=np.float32)

                # next values (bootstrap)
                next_values_list = agent_data['value'][1:] + [0]
                next_values = np.array(next_values_list, dtype=np.float32)

                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values

                advs = []
                adv = 0.0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv)
                advs.reverse()
                advantages = np.array(advs, dtype=np.float32)

                # push sample only if shapes valid and non-empty
                if obs_np.shape[0] == 0 or mask_np.shape[0] == 0 or actions.size == 0:
                    print(f"[{self.name}] Invalid sample for {agent_name}, skipping push.")
                    continue

                self.replay_buffer.push({
                    'state': {
                        'observation': obs_np,
                        'action_mask': mask_np
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })
class ExpertOpponentActor(Actor):
    """
    专家陪练 Actor：
    Player 1 (player_1) 使用当前正在训练的学生模型并记录数据。
    Player 2, 3, 4 使用加载了 expert.pkl 的专家模型作为对手，不记录数据。
    """
    def __init__(self, config, replay_buffer, shared_model_data, expert_model_path):
        super(ExpertOpponentActor, self).__init__(config, replay_buffer, shared_model_data)
        self.expert_model_path = expert_model_path

    def run(self):
        # 优化：在 CPU 上运行采样进程
        torch.set_num_threads(1)
        device = torch.device("cpu")

        # 1. 初始化学生模型 (用于 player_1)
        student_model = CNNModel().to(device)

        # 2. 初始化专家模型 (用于其他 3 个对手)
        expert_model = ResnetModel().to(device)
        try:
            expert_model.load_state_dict(torch.load(self.expert_model_path, map_location=device))
            expert_model.eval()
            print(f"[{self.name}] Expert model loaded successfully from {self.expert_model_path}")
        except Exception as e:
            print(f"[{self.name}] Error loading expert model: {e}")
            return

        # 3. 初始化环境
        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})

        for episode in range(self.config['episodes_per_actor']):
            # 同步学生模型权重
            if self.shared_model_data['version'] > -1:
                student_model.load_state_dict(self.shared_model_data['state_dict'])
            student_model.eval()

            obs = env.reset()
            
            # 初始化存储：我们只需要存储 player_1 的数据
            # 这样可以大幅节省内存并提高 Learner 训练效率
            player_1_data = {
                'state': {'observation': [], 'action_mask': []},
                'action': [], 'reward': [], 'value': []
            }

            done = False
            last_rewards = {n: 0 for n in env.agent_names}

            while not done:
                actions = {}
                
                # 对当前观测中的每个玩家进行决策
                for agent_name, state in obs.items():
                    # 准备张量
                    state_obs = torch.tensor(state['observation'], dtype=torch.float32).unsqueeze(0).to(device)
                    state_mask = torch.tensor(state['action_mask'], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # 构造专家模型和学生模型通用的输入格式
                    input_dict = {
                        'is_training': False, # 采样时均设为 False 以获取确定性策略或处理过的 Logits
                        'obs': {'observation': state_obs, 'action_mask': state_mask}
                    }

                    if agent_name == "player_1":
                        # 学生模型决策
                        with torch.no_grad():
                            # 注意：如果你的 CNNModel forward 不接受 dict，请按需修改
                            logits, value = student_model(input_dict['obs']) 
                            
                            # 应用动作掩码并采样
                            masked_logits = logits.masked_fill(state_mask == 0, -1e9)
                            action_dist = torch.distributions.Categorical(logits=masked_logits)
                            action = int(action_dist.sample().item())
                            
                            # 记录 player_1 的轨迹
                            player_1_data['state']['observation'].append(state['observation'])
                            player_1_data['state']['action_mask'].append(state['action_mask'])
                            player_1_data['action'].append(action)
                            player_1_data['value'].append(float(value.item()))
                    else:
                        # 专家对手决策 (不记录轨迹)
                        with torch.no_grad():
                            # 使用专家模型的 forward 逻辑
                            # 按照你提供的代码，专家模型直接返回 logits，取 argmax
                            expert_logits, _ = expert_model(input_dict)
                            action = int(expert_logits.argmax(dim=-1).item())

                    actions[agent_name] = action

                # 环境步进
                next_obs, rewards, done_dict = env.step(actions)
                
                # 记录 player_1 的奖励
                if "player_1" in rewards:
                    player_1_data['reward'].append(rewards["player_1"])
                
                obs = next_obs
                done = done_dict if isinstance(done_dict, bool) else done_dict.get('__all__', False)

            # --- 战局结束，处理 player_1 的数据并推送到 Buffer ---
            if len(player_1_data['action']) > 0:
                self._push_to_buffer(player_1_data)
                
            if episode % 10 == 0:
                print(f"[{self.name}] Finished Episode {episode}, P1 Reward: {sum(player_1_data['reward'])}")

    def _push_to_buffer(self, data):
        """计算优势函数并将数据推送到共享 Buffer"""
        obs_np = np.stack(data['state']['observation']).astype(np.float32)
        mask_np = np.stack(data['state']['action_mask']).astype(np.float32)
        actions = np.array(data['action'], dtype=np.int64)
        rewards = np.array(data['reward'], dtype=np.float32)
        values = np.array(data['value'], dtype=np.float32)

        # 计算 TD Target 和 GAE 优势
        next_values = np.append(values[1:], 0.0)
        td_target = rewards + self.config['gamma'] * next_values
        td_delta = td_target - values

        advs = []
        adv = 0.0
        for delta in td_delta[::-1]:
            adv = self.config['gamma'] * self.config['lambda'] * adv + delta
            advs.append(adv)
        advs.reverse()
        advantages = np.array(advs, dtype=np.float32)

        # 推送到 ReplayBuffer
        self.replay_buffer.push({
            'state': {'observation': obs_np, 'action_mask': mask_np},
            'action': actions,
            'adv': advantages,
            'target': td_target
        })