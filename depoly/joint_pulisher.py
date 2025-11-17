#!/usr/bin/env python3
import numpy as np
import time
import threading
import socket
import json
from collections import deque
from typing import Callable, Optional, Union

class ExternalDataJointPublisher:
    def __init__(self, frequency: float = 10.0, max_queue_size: int = 1000):
        """
        关节数据发布器，数据从外部传入，支持动态添加
        
        参数:
        - frequency: 发布频率(Hz)，默认10Hz
        - max_queue_size: 最大队列大小，默认1000。当队列满时，新数据会替换最旧的数据
        """
        self.frequency = frequency
        self.interval = 1.0 / frequency  # 发布间隔(秒)
        self.current_index = 0  # 已发送的数据点计数
        self.is_running = False
        self.thread = None
        self.callback = None  # 数据发送回调函数
        self.max_queue_size = max_queue_size
        self.data = deque(maxlen=max_queue_size)  # 使用队列存储数据，支持动态添加，自动限制大小
        self.data_lock = threading.Lock()  # 线程锁，保护数据访问
        self.dropped_count = 0  # 记录因队列满而丢弃的数据点数量
        
        print(f"发布器初始化完成，频率: {frequency}Hz, 间隔: {self.interval:.3f}秒, 最大队列大小: {max_queue_size}")
    
    def set_data(self, data: np.ndarray, clear: bool = True):
        """
        设置关节数据
        
        参数:
        - data: numpy数组，形状为(n, 8)，前7列为关节数据，最后一列为夹具数据
        - clear: 是否清空现有数据后再设置，默认True
        """
        if data is None or data.shape[1] != 8:
            raise ValueError("数据形状应为(n, 8)，前7列为关节数据，最后一列为夹具数据")
        
        with self.data_lock:
            if clear:
                self.data = deque(maxlen=self.max_queue_size)
                self.current_index = 0  # 清空数据时重置计数
                self.dropped_count = 0
            # 将numpy数组转换为队列存储
            for i in range(len(data)):
                was_full = len(self.data) >= self.max_queue_size
                self.data.append(data[i].copy())
                if was_full:
                    self.dropped_count += 1
        
        print(f"数据设置完成，共 {len(self.data)} 个点")
    
    def add_joint(self, point: np.ndarray):
        """
        动态添加单个数据点
        
        参数:
        - point: numpy数组，形状为(8,)，前7个元素为关节数据，最后一个为夹具数据
        """
        if point is None or len(point) != 8:
            raise ValueError("数据点应为长度为8的数组，前7个为关节数据，最后一个为夹具数据")
        
        with self.data_lock:
            # 检查队列是否已满（deque会自动丢弃最旧的数据）
            was_full = len(self.data) >= self.max_queue_size
            self.data.append(point.copy())
            if was_full:
                self.dropped_count += 1
        
        if was_full:
            print(f"警告: 队列已满，最旧数据被丢弃，当前共 {len(self.data)} 个点，已丢弃 {self.dropped_count} 个点")
        else:
            print(f"添加数据点成功，当前共 {len(self.data)} 个点")
    
    def add_joints(self, points: np.ndarray):
        """
        动态添加多个数据点
        
        参数:
        - points: numpy数组，形状为(n, 8)，前7列为关节数据，最后一列为夹具数据
        """
        if points is None or points.shape[1] != 8:
            raise ValueError("数据形状应为(n, 8)，前7列为关节数据，最后一列为夹具数据")
        
        dropped_count_before = self.dropped_count
        with self.data_lock:
            for i in range(len(points)):
                was_full = len(self.data) >= self.max_queue_size
                self.data.append(points[i].copy())
                if was_full:
                    self.dropped_count += 1
        
        dropped = self.dropped_count - dropped_count_before
        if dropped > 0:
            print(f"添加 {len(points)} 个数据点成功，当前共 {len(self.data)} 个点，丢弃了 {dropped} 个旧数据点")
        else:
            print(f"添加 {len(points)} 个数据点成功，当前共 {len(self.data)} 个点")
    
    def clear_data(self):
        """清空所有数据"""
        with self.data_lock:
            count = len(self.data)
            self.data = deque(maxlen=self.max_queue_size)
            self.current_index = 0
            self.dropped_count = 0
        print(f"已清空 {count} 个数据点")
    
    def get_data_count(self) -> int:
        """获取当前数据点数量"""
        with self.data_lock:
            return len(self.data)
    
    def get_queue_stats(self) -> dict:
        """
        获取队列统计信息
        
        返回:
        - dict: 包含队列大小、最大大小、已丢弃数量等信息
        """
        with self.data_lock:
            return {
                'current_size': len(self.data),
                'max_size': self.max_queue_size,
                'dropped_count': self.dropped_count,
                'sent_count': self.current_index,  # 已发送的数据点数量
                'usage_percent': (len(self.data) / self.max_queue_size * 100) if self.max_queue_size > 0 else 0
            }
    
    def set_callback(self, callback: Callable):
        """设置数据发送回调函数"""
        self.callback = callback
    
    def send_to_robot(self) -> bool:
        """从队列头部取出数据并发送到机器人"""
        with self.data_lock:
            if len(self.data) == 0:
                return False
            
            # 从队列头部取出数据（FIFO）
            point = self.data.popleft()
            joint_data = np.array(point[:7])  # 前7个关节，确保是numpy数组
            gripper_data = float(point[7])  # 夹具数据
            point_index = self.current_index  # 当前发送的数据点计数
        
        # 如果有设置回调函数，使用回调函数发送
        if self.callback:
            success = self.callback(point_index, joint_data, gripper_data)
            if success:
                # 发送成功，增加计数
                with self.data_lock:
                    self.current_index += 1
            # 注意：无论成功与否，数据已经从队列中取出，不会重新放回
            return success
        
        # 默认实现：打印数据
        print(f"点 {point_index}: 关节={joint_data}, 夹具={gripper_data}")
        with self.data_lock:
            self.current_index += 1
        return True
    
    def _publishing_loop(self, loop: bool = False):
        """高精度发布循环"""
        with self.data_lock:
            if len(self.data) == 0:
                print("错误: 未设置数据，无法开始发布")
                return
            data_count = len(self.data)
        
        self.is_running = True
        next_time = time.perf_counter()  # 使用高精度计时器
        
        print(f"开始发布关节轨迹，当前共 {data_count} 个点")
        
        consecutive_no_data_count = 0  # 连续没有数据的次数
        max_wait_iterations = 50  # 最大等待次数（约5秒）
        
        while self.is_running:
            # 检查队列是否为空
            with self.data_lock:
                has_more_data = len(self.data) > 0
            
            if not has_more_data:
                if loop:
                    # 循环模式：队列为空，等待新数据
                    consecutive_no_data_count = 0
                    time.sleep(0.1)
                    continue
                else:
                    # 单次模式：等待新数据
                    consecutive_no_data_count += 1
                    if consecutive_no_data_count >= max_wait_iterations:
                        print(f"等待超时，已等待 {max_wait_iterations * 0.1:.1f} 秒，退出发布")
                        break
                    # 等待一下，看是否有新数据动态添加
                    time.sleep(0.1)
                    continue
            else:
                consecutive_no_data_count = 0
            
            # 从队列头部取出数据并发送（send_to_robot内部会popleft并更新current_index）
            success = self.send_to_robot()
            
            if not success:
                # 发送失败（可能是回调函数返回False），但继续下一个点
                print(f"警告: 发送失败，继续下一个点")
                continue
            
            # 计算下一个发布时间点
            next_time += self.interval
            
            # 等待直到下一个发布时间点
            current_time = time.perf_counter()
            sleep_time = next_time - current_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 如果已经超时，调整下一个时间点
                next_time = current_time + self.interval
                print(f"警告: 发布延迟 {-sleep_time:.6f}秒")
        
        self.is_running = False
        print("关节轨迹发布完成")
    
    def start(self, loop: bool = False, async_mode: bool = False):
        """开始发布关节轨迹
        
        参数:
        - loop: 是否循环发布
        - async_mode: 是否异步发布（不阻塞主线程）
        """
        with self.data_lock:
            if len(self.data) == 0:
                print("错误: 未设置数据, 请先调用set_data()或add_joint()方法")
                return
        
        if self.is_running:
            print("发布器已在运行")
            return
        
        self.current_index = 0
        
        if async_mode:
            # 异步模式：在新线程中运行
            self.thread = threading.Thread(target=self._publishing_loop, args=(loop,))
            self.thread.daemon = True
            self.thread.start()
            print("异步发布已启动")
        else:
            # 同步模式：阻塞当前线程
            self._publishing_loop(loop)
    
    def stop(self):
        """停止发布"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("发布已停止")
    
    def get_progress(self) -> tuple:
        """获取发布进度
        
        返回:
        - tuple: (已发送的数据点数量, 队列中剩余的数据点数量)
        """
        with self.data_lock:
            data_count = len(self.data)
            sent_count = self.current_index
        return sent_count, data_count
    
    def is_active(self) -> bool:
        """检查发布器是否正在运行"""
        return self.is_running

# 辅助函数：从文件加载数据
def load_joint_data(file_path: str) -> np.ndarray:
    """
    从文件加载关节数据
    
    参数:
    - file_path: 数据文件路径
    
    返回:
    - numpy数组, 形状为(n, 8)
    """
    try:
        data = np.load(file_path)
        if data.shape[1] != 8:
            raise ValueError(f"数据形状应为(n, 8)，但实际为{data.shape}")
        print(f"从 {file_path} 加载数据成功，形状: {data.shape}")
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise

# Socket通信回调函数
def create_socket_callback(host: str, port: int, timeout: float = 5.0):
    """
    创建Socket通信回调函数
    
    参数:
    - host: 目标主机
    - port: 目标端口
    - timeout: 连接超时时间(秒)
    
    返回:
    - 回调函数和socket对象
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        sock.connect((host, port))
        print(f"成功连接到机器人: {host}:{port}")
    except Exception as e:
        print(f"连接失败: {e}")
        sock = None
    
    def socket_callback(point_index, joint_data, gripper_data):
        if sock is None:
            return False
        
        # 格式化数据为JSON
        message = {
            'timestamp': time.time(),
            'index': point_index,
            'joints': joint_data.tolist(),
            'gripper': float(gripper_data)
        }
        
        try:
            # 发送数据
            sock.send((json.dumps(message) + '\n').encode())
            print(f"发送第 {point_index} 个关节点")
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            return False
    
    return socket_callback, sock

# UDP通信回调函数
def create_udp_callback(host: str, port: int):
    """
    创建UDP通信回调函数
    
    参数:
    - host: 目标主机
    - port: 目标端口
    
    返回:
    - 回调函数
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def udp_callback(point_index, joint_data, gripper_data):
        # 格式化数据为JSON
        message = {
            'timestamp': time.time(),
            'index': point_index,
            'joints': joint_data.tolist(),
            'gripper': float(gripper_data)
        }
        
        try:
            # 发送数据
            sock.sendto(json.dumps(message).encode(), (host, port))
            print(f"UDP发送第 {point_index} 个关节点")
            return True
        except Exception as e:
            print(f"UDP发送失败: {e}")
            return False
    
    return udp_callback, sock

# 使用示例
if __name__ == "__main__":
    # 创建发布器实例，设置最大队列大小为100
    publisher = ExternalDataJointPublisher(frequency=100, max_queue_size=1000)  # 10Hz, 最大队列100个点
    
    try:
        # 示例1: 从文件加载数据（传统方式）
        data = load_joint_data('data/trajectory_1761875980.4997294.npy')  # 替换为您的数据文件路径
        publisher.add_joints(data)

        print(f"data: {data}")

        # 示例4: 使用Socket通信
        print("=== 尝试连接Socket服务器 ===")
        socket_callback, sock = create_socket_callback('192.168.100.100', 12345)
        if sock is not None:
            publisher.set_callback(socket_callback)
            print("Socket回调已设置")
        else:
            print("警告: Socket连接失败,将使用默认打印模式")
        
        # # 示例2: 动态添加数据点
        # print("\n=== 添加初始数据点 ===")
        # # 添加单个点
        # point1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # publisher.add_joint(point1)
        
        # # 添加多个点
        # points = np.array([
        #     [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # ])
        # publisher.add_joints(points)
        # print(f"初始数据点数量: {publisher.get_data_count()}")
        
        # # 示例3: 异步发布并动态添加数据
        # print("\n=== 启动异步发布，支持动态添加数据 ===")
        publisher.start(loop=False, async_mode=True)
        
        # # 等待一下，确保发布器已启动
        # time.sleep(0.2)
        
        # # 主线程可以继续添加数据
        # print("\n=== 开始动态添加数据点 ===")
        # time.sleep(0.5) 
        # for i in range(20):
        #      # 模拟数据生成间隔
        #     time.sleep(0.1) 
        #     new_point = np.array([0.1*i, 0.2*i, 0.3*i, 0.4*i, 0.5*i, 0.6*i, 0.7*i, 0.8*i])
        #     publisher.add_joint(new_point)
        #     progress = publisher.get_progress()
        #     stats = publisher.get_queue_stats()
        #     print(f"已添加第 {i+1} 个新点，当前进度: {progress[0]}/{progress[1]}, "
        #           f"队列使用率: {stats['usage_percent']:.1f}%, 已丢弃: {stats['dropped_count']}")

        # time.sleep(2.0) 
        # for i in range(20):
        #      # 模拟数据生成间隔
        #     time.sleep(0.1) 
        #     new_point = np.array([0.1*i, 0.2*i, 0.3*i, 0.4*i, 0.5*i, 0.6*i, 0.7*i, 0.8*i])
        #     publisher.add_joint(new_point)
        #     stats = publisher.get_queue_stats()
        #     if stats['dropped_count'] > 0:
        #         print(f"队列统计: 使用率 {stats['usage_percent']:.1f}%, 已丢弃 {stats['dropped_count']} 个点")
        #     progress = publisher.get_progress()
        #     print(f"已添加第 {i+1} 个新点，当前进度: {progress[0]}/{progress[1]}")
        
        print("\n=== 等待发布完成 ===")
        # 等待发布完成
        while publisher.is_active():
            progress = publisher.get_progress()
            print(f"发布进度: {progress[0]}/{progress[1]}")
            time.sleep(1)
        

        
        # 示例5: 使用UDP通信
        # udp_callback, udp_sock = create_udp_callback('192.168.1.100', 12346)
        # publisher.set_callback(udp_callback)
        
        # 示例6: 同步发布一次
        # print("=== 同步发布一次 ===")
        # publisher.start(loop=False, async_mode=False)
        
        # 示例7: 异步循环发布
        # print("=== 异步循环发布 ===")
        # publisher.start(loop=True, async_mode=True)
        
    except KeyboardInterrupt:
        print("\n关节轨迹发布被中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        publisher.stop()
        # 如果使用Socket，记得关闭
        if 'sock' in locals() and sock is not None:
            sock.close()
            print("Socket连接已关闭")
        # if 'udp_sock' in locals() and udp_sock:
        #     udp_sock.close()