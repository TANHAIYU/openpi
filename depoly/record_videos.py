import rospy
import os
import json
import cv2
import numpy as np
from datetime import datetime
from sensor_msgs.msg import CompressedImage, JointState
import logging

# ================= 配置参数 =================
SAVE_DIR = f"/home/abc/Documents/recorded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 存储主目录
VIDEO_FPS = 30  # 视频帧率（根据实际订阅频率调整）
VIDEO_CODEC = 'mp4v'  # 视频编码（mp4格式推荐）
VIDEO_EXT = '.mp4'  # 视频文件后缀
JOINT_STATE_FILENAME = "joint_states.jsonl"  # 关节状态文件
VERBOSE_LOG = True  # 详细日志开关

# ================= 全局变量 =================
# 视频写入器字典（key：相机名称，value：cv2.VideoWriter）
video_writers = {}
# 记录每个相机的图像尺寸（首次接收图像时初始化）
camera_sizes = {}

# ================= 日志配置 =================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ================= 初始化存储目录 =================
def init_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        logger.info(f"创建存储目录: {SAVE_DIR}")
    
    # 关节状态文件初始化
    joint_state_path = os.path.join(SAVE_DIR, JOINT_STATE_FILENAME)
    with open(joint_state_path, 'a') as f:
        pass
    logger.info(f"关节状态将保存到: {joint_state_path}")

# ================= 视频写入器初始化 =================
def init_video_writer(camera_name, frame_size):
    """初始化指定相机的视频写入器"""
    video_filename = f"{camera_name}_camera{VIDEO_EXT}"
    video_path = os.path.join(SAVE_DIR, video_filename)
    
    # 获取编码ID
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    
    # 创建VideoWriter对象（路径、编码、帧率、图像尺寸）
    writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, frame_size)
    
    if not writer.isOpened():
        logger.error(f"[{camera_name}] 视频写入器初始化失败！")
        return None
    
    video_writers[camera_name] = writer
    logger.info(f"[{camera_name}] 视频写入器初始化成功，保存路径: {video_path}")
    return writer

# ================= 回调函数（保存视频+关节状态） =================
def camera_callback(msg, camera_name):
    """相机图像回调：解码并写入视频"""
    try:
        # 解码压缩图像
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning(f"[{camera_name}] 图像解码失败")
            return
        
        # 获取图像尺寸（高x宽x通道）
        frame_size = (image.shape[1], image.shape[0])  # VideoWriter需要（宽，高）格式
        
        # 首次接收图像时初始化视频写入器
        if camera_name not in video_writers:
            init_video_writer(camera_name, frame_size)
        
        # 写入视频帧
        writer = video_writers.get(camera_name)
        if writer and writer.isOpened():
            writer.write(image)
            if VERBOSE_LOG:
                logger.debug(f"[{camera_name}] 写入视频帧 (尺寸: {frame_size})")
        else:
            logger.error(f"[{camera_name}] 视频写入器未就绪，跳过该帧")
    
    except Exception as e:
        logger.error(f"[{camera_name}] 视频写入失败: {str(e)}")

def state_callback(msg, topic_name):
    """关节状态回调：保存为JSONL格式"""
    try:
        state_data = {
            "topic": topic_name,
            "timestamp": msg.header.stamp.to_sec(),
            "timestamp_str": datetime.fromtimestamp(msg.header.stamp.to_sec()).strftime('%Y%m%d_%H%M%S_%f'),
            "joint_names": msg.name,
            "positions": msg.position,
            "velocities": msg.velocity if msg.velocity else [],
            "efforts": msg.effort if msg.effort else []
        }
        
        joint_state_path = os.path.join(SAVE_DIR, JOINT_STATE_FILENAME)
        with open(joint_state_path, 'a', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False)
            f.write('\n')
        
        if VERBOSE_LOG:
            logger.debug(f"[{topic_name}] 保存关节状态: {state_data['timestamp_str']}")
    
    except Exception as e:
        logger.error(f"[{topic_name}] 关节状态保存失败: {str(e)}")

# ================= 批量创建订阅者 =================
def setup_subscribers():
    subscriptions = [
        # 相机话题
        ("/cam/head/color/image_raw/compressed", CompressedImage, camera_callback, "head"),
        ("/cam/right_arm/wrist/color/image_raw/compressed", CompressedImage, camera_callback, "right_arm"),
        ("/cam/left_arm/wrist/color/image_raw/compressed", CompressedImage, camera_callback, "left_arm"),
        # 关节状态话题
        ("/left_arm/joint_states", JointState, state_callback, "/left_arm/joint_states"),
        ("/left_arm_gripper/joint_states", JointState, state_callback, "/left_arm_gripper/joint_states"),
    ]
    
    subscribers = []
    for topic, msg_type, callback, extra_arg in subscriptions:
        if callback == camera_callback:
            logger.info(f"订阅相机话题: {topic} (将保存为 {extra_arg}_camera{VIDEO_EXT})")
        else:
            logger.info(f"订阅关节状态话题: {topic}")
        
        sub = rospy.Subscriber(
            topic,
            msg_type,
            lambda msg, arg=extra_arg, cb=callback: cb(msg, arg),
            queue_size=5  # 增大队列，避免高帧率时丢帧
        )
        subscribers.append(sub)
    
    return subscribers

# ================= 资源清理 =================
def cleanup_resources():
    """关闭视频写入器，释放资源"""
    logger.info("开始清理资源...")
    
    # 关闭所有视频写入器
    for camera_name, writer in video_writers.items():
        if writer.isOpened():
            writer.release()
            logger.info(f"[{camera_name}] 视频写入器已关闭")
    
    logger.info(f"所有数据已保存到: {os.path.abspath(SAVE_DIR)}")
    logger.info("录制完成！")

# ================= 主函数 =================
if __name__ == "__main__":
    # 初始化存储目录
    init_save_dir()
    
    # 初始化ROS节点
    rospy.init_node("topic_video_recorder", anonymous=True)
    logger.info("ROS节点启动：开始录制视频和关节状态...")
    
    # 设置订阅者
    subscribers = setup_subscribers()
    
    # 注册关闭回调（确保程序退出时关闭视频写入器）
    rospy.on_shutdown(cleanup_resources)
    
    # 保持节点运行
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        logger.info("收到关闭信号，正在停止录制...")
    finally:
        cleanup_resources()