# ros_agent.py
import os
import sys
import time
import math
import json
import logging
import numpy as np
import cv2
import rospy
import websocket
from pathlib import Path
from cv_bridge import CvBridge
from openpi_client import msgpack_numpy
from sensor_msgs.msg import CompressedImage, JointState
sys.path.append("/home/abc/dev/galbot_control_interface")
from galbot_control_interface import GalbotControlInterface


from joint_pulisher import ExternalDataJointPublisher
import joint_pulisher
import logging


# ---------------------- CONFIG ----------------------
CAMERA_NAMES = ["head", "left_arm", "right_arm"]
GRIPPER_OPEN = 0.65
GRIPPER_CLOSE = 0.0
DECAY_RATE = 3.0

LEFT_ARM_RESET = [
    1.0093586444854736, -1.3492074012756348, -1.134690761566162,
    -1.9343969821929932, -0.1084338054060936, 0.2987898588180542, 0.3599584102630615
]

DEPLOY_CONFIG = {
    "max_steps": 200,
    "infer_freq": 20,
    "output_dir": "/home/abc/galbot_records/5",
    "ws_url": "ws://127.0.0.1:8000"
}

STEP_ACTION_HORIZON_MAX = 50
STEP_ACTION_HORIZON_MIN = 5


# ---------------------- ROS 数据缓存 ----------------------
BRIDGE = CvBridge()
CAMERA_IMAGES = {k: None for k in CAMERA_NAMES}
STATE = {
    "/left_arm/joint_states": None,
    "/left_arm_gripper/joint_states": None
}


def camera_callback(msg: CompressedImage, cam_name: str):
    try:
        img = BRIDGE.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        CAMERA_IMAGES[cam_name] = img
    except Exception as e:
        rospy.logerr(f"Camera {cam_name} callback error: {e}")


def state_callback(msg: JointState, state_name: str):
    try:
        STATE[state_name] = list(msg.position)
    except Exception as e:
        rospy.logerr(f"State {state_name} callback error: {e}")


def make_observation() -> dict | None:
    """构造符合 OpenPI 输入格式的 obs, 保持 numpy.ndarray 类型"""

    missing_cams = [cam for cam in CAMERA_NAMES if CAMERA_IMAGES.get(cam) is None]
    if missing_cams:
        rospy.logwarn(f"Missing camera images: {missing_cams}. Skipping this observation.")
        return None

    left_arm = STATE.get("/left_arm/joint_states")
    gripper = STATE.get("/left_arm_gripper/joint_states")
    if left_arm is None or gripper is None:
        rospy.logwarn("Missing joint states. Skipping this observation.")
        return None

    images = {}
    for cam in CAMERA_NAMES:
        try:
            images[cam] = cv2.cvtColor(CAMERA_IMAGES[cam], cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Error converting camera {cam} image to RGB: {e}")
            return None

    # 4️⃣ 拼接关节状态（7 左臂 + 1 gripper 或更多）
    joint_state = np.concatenate([np.array(left_arm).flatten(), np.array(gripper).flatten()])

    # 5️⃣ 构造 observation 字典（保持 numpy.ndarray 类型）
    obs = {
        "state": joint_state,             # numpy.ndarray
        "image": images["head"],          # numpy.ndarray
        "wrist_image_left": images["left_arm"],# numpy.ndarray
        "wrist_image_right": images["right_arm"], # numpy.ndarray
        "prompt": "place the spider board in the designated position."
    }

    return obs


# -------------------------- Robotic Arm Control Wrapper --------------------------
# UDP通信回调函数
import socket

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


# ---------------------- Galbot 控制 ----------------------
class GalbotController:
    def __init__(self, logger):
        self.interface = GalbotControlInterface(log_level="error")
        self.publisher = ExternalDataJointPublisher(frequency=50, max_queue_size=1000)  # 10Hz, 最大队列100个点
        socket_callback, sock = create_socket_callback('192.168.100.100', 12345)
        self.start_pulish = False
        if sock is not None:
            self.publisher.set_callback(socket_callback)
            print("Socket回调已设置")
        else:
            print("警告: Socket连接失败,将使用默认打印模式")
        self.logger = logger
        # self.reset_pose()

    def reset_pose(self):
        self.interface.set_arm_joint_angles(
            arm_joint_angles=LEFT_ARM_RESET, 
            speed=1.0, 
            arm="left_arm", 
            asynchronous=True
        )
        time.sleep(2)
    
    def open_gripper(self):
        self.interface.set_gripper_status(
            width_percent=GRIPPER_OPEN, 
            speed=1.0, 
            force=10, 
            gripper="left_gripper"
        )

    def execute_action(self, action: np.ndarray):
        """Execute actions output by OpenPI (7 joints + 1 gripper)"""
        # joint actions (first 7 dimensions)
        arm_joints = action[:7].tolist()
        arm_status = self.interface.set_arm_joint_angles(
            arm_joint_angles=arm_joints, 
            speed=1, 
            arm="left_arm", 
            asynchronous=True
        )
        
        # gripper action (8th dimension, normalized to [0,1])
        gripper_pos = np.clip((action[7]-0.002)/0.058, 0.0, 0.65)
        gripper_status = self.interface.set_gripper_status(
            width_percent=gripper_pos, 
            speed=1.0, 
            force=11, 
            gripper="left_gripper"
        )
        return arm_status, gripper_status
    
    def execute_action_realtime(self, actions: np.ndarray):
        if isinstance(actions, np.ndarray):
            actions[:, 7] = np.maximum(actions[:, 7] - 0.004, 0.0)
            actions[:, 7] = np.minimum(actions[:, 7], 0.045)
            # actions[:, 5] = np.maximum(actions[:, 5], 0.736)  # limit wrist joint


        self.publisher.add_joints(actions)

        if not self.start_pulish:
            self.publisher.start(loop=False, async_mode=True)
            self.start_pulish = True


# ---------------------- 推理客户端 ----------------------
class WebSocketPolicyClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.connect()

    def connect(self):
        rospy.loginfo(f"Connecting to model server at {self.ws_url} ...")
        self.ws = websocket.create_connection(self.ws_url, timeout=10)
        rospy.loginfo("Connected to model server ✅")

    def infer(self, obs: dict):
        """发送 obs 到 model_server, 并返回推理结果"""
        try:
            # msgpack 序列化
            payload = msgpack_numpy.packb(obs, use_bin_type=True)
            self.ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)

            # 接收二进制消息
            result = self.ws.recv()

            # 确保 bytes 类型
            if isinstance(result, bytes):
                result = msgpack_numpy.unpackb(result, raw=False)
            else:
                rospy.logerr(f"Unexpected result type: {type(result)}, result: {result}")
                return None

            return result

        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
            # 尝试重连
            try:
                self.connect()
            except Exception as e2:
                rospy.logerr(f"Reconnect failed: {e2}")
            return None

def run_inference_loop(policy_client):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    controller = GalbotController(logger)
    step = 1

    while step < DEPLOY_CONFIG["max_steps"] and not rospy.is_shutdown():
        time.sleep(0.1)
        obs = make_observation()
        if obs is None:
            time.sleep(0.05)
            continue

        # ------------------------------
        # 1. 先 flush 掉 WebSocket 的旧消息（关键）
        # ------------------------------
        try:
            policy_client.ws.settimeout(0.001)
            while True:
                try:
                    _ = policy_client.ws.recv()
                except:
                    break
        finally:
            policy_client.ws.settimeout(2)

        # ------------------------------
        # 2. 正式开始计时 + 调 infer
        # ------------------------------
        start_time = time.time()
        result = policy_client.infer(obs)

        if result is None or len(result.keys()) == 0:
            rospy.logwarn("Inference failed, skipping this step.")
            continue

        infer_time = (time.time() - start_time) * 1000
        rospy.loginfo(f"Inference latency: {infer_time:.1f} ms")

        # ------------------------------
        # 3. 执行动作
        # ------------------------------
        actions = np.array(result["actions"])
        controller.execute_action_realtime(actions[:15])

        time.sleep(0.5)
        rospy.loginfo(f"---- executing step {step} actions ----")
        step += 1


# ---------------------- 主函数 ----------------------
def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    rospy.init_node("openpi_ros_agent", anonymous=True)

    rospy.loginfo("Starting OpenPI ROS Agent with WebSocket Model Server...")
    for cam in CAMERA_NAMES:
        topic = f"/cam/{cam}/wrist/color/image_raw/compressed" if "arm" in cam else f"/cam/{cam}/color/image_raw/compressed"
        rospy.Subscriber(topic, CompressedImage, lambda msg, c=cam: camera_callback(msg, c), queue_size=1)
        rospy.loginfo(f"Subscribed to {topic}")

    rospy.Subscriber("/left_arm/joint_states", JointState,
                     lambda msg, s="/left_arm/joint_states": state_callback(msg, s), queue_size=1)
    rospy.Subscriber("/left_arm_gripper/joint_states", JointState,
                     lambda msg, s="/left_arm_gripper/joint_states": state_callback(msg, s), queue_size=1)

    time.sleep(1)
    client = WebSocketPolicyClient(DEPLOY_CONFIG["ws_url"])
    run_inference_loop(client)


if __name__ == "__main__":
    main()
