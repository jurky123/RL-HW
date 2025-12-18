# train_launcher.py (替代你原来的 main 脚本)
from replay_buffer import ReplayBuffer
from actor import Actor,ExpertOpponentActor
from learner import Learner
from multiprocessing import Manager
import cProfile
import pstats
import yappi
import os
import torch
from replay_buffer import ReplayBuffer
# ... 其他导入保持不变

def main():
    # --- 1. 配置参数 (参数合理性建议见后文) ---
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
        'ckpt_save_path': 'checkpoint/',
        'resume_training': True,  # 新增：断点续传开关
        'latest_model_path': 'checkpoint/latest_model.pkl' # 最新模型路径
    }

    manager = Manager()
    
    # --- 2. 核心修改：检查并加载历史权重 ---
    initial_state_dict = None
    initial_version = -1

    if config['resume_training'] and os.path.exists(config['latest_model_path']):
        print(f">>> 发现历史模型: {config['latest_model_path']}，正在恢复训练...")
        try:
            # 兼容性处理：如果保存的是整个文件或 state_dict
            checkpoint = torch.load(config['latest_model_path'], map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                initial_state_dict = checkpoint['state_dict']
                initial_version = checkpoint.get('version', 0)
            else:
                initial_state_dict = checkpoint
                initial_version = 0
            print(f">>> 恢复成功，当前模型版本: {initial_version}")
        except Exception as e:
            print(f">>> 恢复失败: {e}，将开始全新训练")

    # 初始化共享内存
    shared_model_data = manager.dict({
        'version': initial_version,
        'state_dict': initial_state_dict  # Actor 启动后会立即拉取这个数据
    })
    # ---------------------------------------

    # 后续启动代码保持不变...
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    # ... (启动 actors 和 learner)
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