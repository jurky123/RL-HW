from multiprocessing import Process
import numpy as np
import torch
import time # 引入 time 用于等待初始模型

from replay_buffer import ReplayBuffer
# from model_pool import ModelPoolClient  # <-- 移除 ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

class Actor(Process):
    
    # 接收 shared_model_data 字典
    def __init__(self, config, replay_buffer, shared_model_data):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.shared_model_data = shared_model_data  # <-- 新增共享字典属性
        
    def run(self):
        torch.set_num_threads(1)
        
        # # connect to model pool  <-- 移除 ModelPoolClient 连接逻辑
        
        # create network model
        model = CNNModel()
        
        # load initial model  <-- 修改为从共享字典加载
        version = {'id': -1}
        
        # 循环等待 Learner 推送第一个模型 (version != -1)
        print(f"[{self.name}] Waiting for initial model from Learner...")
        while self.shared_model_data['version'] == -1:
            time.sleep(0.1)
        
        latest_version_id = self.shared_model_data['version']
        state_dict = self.shared_model_data['state_dict']
        model.load_state_dict(state_dict)
        version['id'] = latest_version_id
        print(f"[{self.name}] Loaded initial model version {latest_version_id}")
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        for episode in range(self.config['episodes_per_actor']):
            # update model  <-- 修改为从共享字典更新
            latest_id = self.shared_model_data['version']
            
            # 检查是否有新版本
            if latest_id > version['id']:
                state_dict = self.shared_model_data['state_dict']
                model.load_state_dict(state_dict)
                version['id'] = latest_id
                
            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    
                    # 确保 state['observation'] 和 state['action_mask'] 是 PyTorch 张量
                    state_obs = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    state_mask = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                    state_tensor = {'observation': state_obs, 'action_mask': state_mask}
                    
                    model.train(False) # Batch Norm inference mode
                    with torch.no_grad():
                        logits, value = model(state_tensor) # 使用新的 tensor state
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                    
                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs
                
            # 打印日志时使用 version['id']
            print(self.name, 'Episode', episode, 'Model', version['id'], 'Reward', rewards) 
            
            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                obs = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                actions = np.array(agent_data['action'], dtype = np.int64)
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                values = np.array(agent_data['value'], dtype = np.float32)
                
                # 如果是最后一个状态，下一个状态的 value 为 0
                next_values_list = agent_data['value'][1:] + [0]
                next_values = np.array(next_values_list, dtype = np.float32)
                
                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })