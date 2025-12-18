from multiprocessing import Queue
import numpy as np
import random
import sys
from typing import Dict, Any, List # 引入类型提示

class ReplayBuffer:
    def __init__(self, capacity: int, max_queue_size: int = 10):
        """
        capacity: 最大存储样本数量
        max_queue_size: 队列最大长度（控制actor非阻塞写入）
        """
        self.queue = Queue(maxsize=max_queue_size)
        self.capacity = capacity
        
        # -------------------- 核心变量 --------------------
        self.buffer: Dict[str, Any] = {} # 存储扁平化后的 NumPy 数组
        self.ptr = 0 # 写入指针 (Write Pointer)
        self.size_ = 0 # 当前已存储的样本总数
        
        # 记录 Episode 边界，以便于区分有效数据和已覆盖的数据 (可选，但推荐保留)
        # self.episode_start_indices = [] 
        # ------------------------------------------------
        
        self.stats = {'sample_in': 0, 'sample_out': 0, 'episode_in': 0}

    # -------------------- Actor 写入 --------------------
    def push(self, episode_data: Dict[str, Any]):
        try:
            # 演员将完整的 Episode (Dict of np.array) 推入队列
            self.queue.put_nowait(episode_data)
        except Exception:
            pass 

    # -------------------- Learner 存储 (新的 _flush 逻辑) --------------------
    def _initialize_buffer(self, data: Dict[str, Any]):
        """根据第一个收到的 Episode 数据初始化 Buffer 数组"""
        # 递归遍历数据结构
        def init_rec(buf: Dict[str, Any], data_to_init: Dict[str, Any]):
            for k, v in data_to_init.items():
                if isinstance(v, dict):
                    buf[k] = {}
                    init_rec(buf[k], v)
                else:
                    # 假定 v 是 np.array，获取其形状和 dtype
                    sample_shape = v.shape[1:] 
                    sample_dtype = v.dtype
                    
                    # 预分配巨大的扁平化数组，只执行一次
                    buf[k] = np.zeros((self.capacity, *sample_shape), dtype=sample_dtype)
                    print(f"[Buffer Init] Allocated array for '{k}' with shape {buf[k].shape} and dtype {sample_dtype}")
        
        init_rec(self.buffer, data)

    def _store(self, data: Dict[str, Any], start_index: int, length: int):
        """将数据直接写入预分配的数组中"""
        
        def store_rec(buf: Dict[str, Any], data_to_store: Dict[str, Any], start: int, length: int):
            # 处理循环写入：如果数据块跨越 buffer 尾部 (capacity) 到头部 (0)
            end = start + length
            if end > self.capacity:
                # 跨越边界：分成两部分写入
                split_point = self.capacity - start
                
                # Part 1: 写入到 buffer 尾部
                for k, v in data_to_store.items():
                    if isinstance(v, dict):
                        store_rec(buf[k], v, start, split_point)
                    else:
                        buf[k][start:self.capacity] = v[:split_point]

                # Part 2: 写入到 buffer 头部
                remaining_length = length - split_point
                for k, v in data_to_store.items():
                    if isinstance(v, dict):
                        store_rec(buf[k], v, 0, remaining_length)
                    else:
                        buf[k][0:remaining_length] = v[split_point:]
            else:
                # 非跨越边界：直接写入
                for k, v in data_to_store.items():
                    if isinstance(v, dict):
                        store_rec(buf[k], v, start, length)
                    else:
                        buf[k][start:end] = v

        store_rec(self.buffer, data, start_index, length)
    
    
    def _flush(self):
        # 第一次调用时，等待第一个 Episode 初始化 Buffer
        if not self.buffer:
            try:
                # 尝试获取第一个 Episode 数据来初始化
                first_episode = self.queue.get(timeout=0.1)
                self._initialize_buffer(first_episode)
                self.queue.put_nowait(first_episode) # 放回去，确保它也会被处理
            except:
                return # 队列仍然为空，无法初始化

        while True:
            try:
                episode_data = self.queue.get_nowait()
                length = self._episode_length(episode_data)

                # 检查是否超过 capacity (一般只在 capacity 极小时发生)
                if length > self.capacity: continue

                # 写入数据
                self._store(episode_data, self.ptr, length)
                
                # 更新指针和 size
                self.ptr = (self.ptr + length) % self.capacity
                self.size_ = min(self.size_ + length, self.capacity)

                self.stats['sample_in'] += length
                self.stats['episode_in'] += 1
            except Exception:
                break # 队列空

    def _episode_length(self, data: Dict[str, Any]):
        """获取 episode 样本长度（辅助函数，沿用旧逻辑）"""
        if isinstance(data, dict):
            for v in data.values():
                return self._episode_length(v)
        else:
            return len(data)

    # -------------------- 采样 (核心优化) --------------------
    def _get_samples_by_indices(self, buf: Dict[str, Any], indices: np.ndarray):
        """递归地根据 NumPy 索引从扁平数组中提取样本"""
        res = {}
        for k, v in buf.items():
            if isinstance(v, dict):
                res[k] = self._get_samples_by_indices(v, indices)
            else:
                # 直接使用索引数组进行切片，无需 np.concatenate
                res[k] = v[indices]
        return res
        
    def sample(self, batch_size: int):
        self._flush() 
        
        total_samples = self.size_ # 直接使用 size_ 成员
        if total_samples == 0:
            return None

        actual_batch_size = min(batch_size, total_samples)
        
        # 1. 随机选择 batch_size 个索引，从 0 到 size_-1
        # indices 对应于 buffer 中有效的样本位置
        indices = np.random.choice(total_samples, actual_batch_size, replace=False)

        # 2. 转换索引: 逻辑索引 -> 物理索引 (ptr 是下一个写入位置)
        # 由于写入是循环的，有效数据是从 ptr 到 capacity-1, 
        # 然后从 0 到 ptr-1。
        
        # **简化循环缓冲区索引**
        # 在一个全满的循环数组中，ptr 指向最旧的数据。
        # 当 size_ < capacity 时，有效索引就是 0 到 size_-1。
        # 当 size_ == capacity 时，有效索引就是 0 到 capacity-1。
        
        # 物理索引 = (ptr + 逻辑索引) % capacity
        
        # 如果 buffer 未满，逻辑索引 = 物理索引
        if self.size_ < self.capacity:
            physical_indices = indices
        # 如果 buffer 已满，ptr 处是最旧的样本，逻辑 0 应该是 (ptr)
        else:
            # 物理索引 = (ptr + 逻辑索引) % capacity
            physical_indices = (self.ptr + indices) % self.capacity
        
        # 3. 提取批次数据
        batch = self._get_samples_by_indices(self.buffer, physical_indices)
        
        self.stats['sample_out'] += actual_batch_size
        return batch

    # -------------------- 其它接口 --------------------
    def size(self):
        self._flush()
        return self.size_

    def clear(self):
        self.buffer = {}
        self.ptr = 0
        self.size_ = 0
        self.stats = {'sample_in': 0, 'sample_out': 0, 'episode_in': 0}
        
    # **注意：_flatten_buffer, _total_len_recursive, _trim_buffer 等函数已不再需要，已移除。**