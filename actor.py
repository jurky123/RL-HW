from multiprocessing import Process
import numpy as np
import torch
import time

from replay_buffer import ReplayBuffer
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

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
