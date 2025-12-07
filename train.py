# train_launcher.py (替代你原来的 main 脚本)
from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from multiprocessing import Manager

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 10000,
        'replay_buffer_episode': 200,
        'num_actors': 128,
        'episodes_per_actor': 10000,
        'gamma': 0.99,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 2048,
        'epochs': 10,
        'clip': 0.15,
        'lr': 3e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.03,
        'device': 'cuda',
        'ckpt_save_interval': 120,
        'ckpt_save_path': '/home/jiayu/models/wan/RLHW/RL-HW/checkpoint/'
    }

    # Manager: 用于共享 replay_buffer 和 shared_model_data（Actor 与 Learner 进程间）
    manager = Manager()

    shared_model_data = manager.dict({
        'version': -1,
        'state_dict': None
    })

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])

    # 启动 actors
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer, shared_model_data)
        actors.append(actor)

    # 启动 learner（作为一个进程）
    learner = Learner(config, replay_buffer, shared_model_data)

    # 启动进程
    for actor in actors:
        actor.start()
    learner.start()

    # 等待 actors 结束（通常不会直接结束），并在结束后杀 learner
    for actor in actors:
        actor.join()

    # 如果 actors 全结束，终止 learner
    learner.terminate()
