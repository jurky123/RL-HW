from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import time

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 24,
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cpu',
        'ckpt_save_interval': 300,
        'ckpt_save_path': '/model/'
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actor.daemon = False  # 确保是非守护进程
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    learner.daemon = False
    
    # 启动所有进程
    for actor in actors: 
        actor.start()
    learner.start()
    
    # 方案1：使用超时等待
    print("等待所有Actor完成...")
    timeout = config['episodes_per_actor'] * 60  # 估计的总超时时间（秒）
    start_time = time.time()
    
    for i, actor in enumerate(actors):
        remaining_timeout = max(1, timeout - (time.time() - start_time))
        print(f"等待 {actor.name}，剩余超时时间：{remaining_timeout:.1f}s")
        actor.join(timeout=remaining_timeout)
        if actor.is_alive():
            print(f"警告：{actor.name} 超时，强制终止")
            actor.terminate()
            actor.join(timeout=5)
            if actor.is_alive():
                actor.kill()
    
    # 终止Learner
    print("终止Learner进程...")
    learner.terminate()
    learner.join(timeout=5)
    if learner.is_alive():
        learner.kill()
    
    print("训练完成")