# train_launcher.py (替代你原来的 main 脚本)
from replay_buffer import ReplayBuffer
from actor import Actor,ExpertOpponentActor
from learner import Learner
from multiprocessing import Manager
import cProfile
import pstats
import yappi
import os
def main():
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 2000,
        'num_actors': 8,
        'num_expert_actors': 4,
        'expert_model_path': 'checkpoint/expert.pkl',
        'episodes_per_actor': 10000,
        'gamma': 0.99,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 128,
        'epochs': 10,
        'clip': 0.15,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.03,
        'device': 'cuda',
        'ckpt_save_interval': 600,
        'ckpt_save_path': 'checkpoint/'
    }

    # Manager: 用于共享 replay_buffer 和 shared_model_data（Actor 与 Learner 进程间）
    manager = Manager()
    if not os.path.exists(config['expert_model_path']):
        print(f"警告: 找不到专家模型 {config['expert_model_path']}，请检查路径！")
        # 如果没有专家模型，可以将 num_expert_actors 设为 0

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
    #学生actor
    for i in range(config['num_expert_actors']):
        local_config = config.copy()
        local_config['name'] = 'ExpertOppo-Actor-%d' % i
        # 使用 ExpertOpponentActor 类
        actor = ExpertOpponentActor(
            local_config, 
            replay_buffer, 
            shared_model_data, 
            config['expert_model_path']
        )
        actors.append(actor)
    #专家actor

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
    return
if __name__ == '__main__':
    """yappi.set_clock_type('cpu')   # 或 'wall'
    yappi.start()"""
    main()  # 你原来的 run 方法
    """yappi.stop()
    yappi.get_func_stats().print_all()  # 打印到 stdout"""
#--standalone --nproc_per_node=4 models/wan/RLHW/RL-HW/train.py