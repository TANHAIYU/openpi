# ros_agent.py
import os
import sys
import time
import math
import json
import logging
import argparse
import yaml
import numpy as np
import cv2
import rospy
import websocket
import socket
from pathlib import Path
from enum import Enum, auto
from cv_bridge import CvBridge
from openpi_client import msgpack_numpy
from sensor_msgs.msg import CompressedImage, JointState
from galbot_control_interface import GalbotControlInterface

from joint_pulisher import ExternalDataJointPublisher

CFG = {}

class TaskStage(Enum):
    PLACING = auto()
    PLACED = auto()
    FAILED = auto()

class TaskStateMachine:
    def __init__(self, threshold):
        self.current_state = TaskStage.PLACING
        self.gripper_threshold = threshold 

    def update(self, actions: np.ndarray):
        if self.current_state == TaskStage.PLACING:
            last_gripper_val = actions[-1, 7]
            if last_gripper_val > self.gripper_threshold:
                rospy.loginfo(f"State Transition: PLACING -> PLACED (Val: {last_gripper_val:.4f})")
                self.current_state = TaskStage.PLACED

    def is_placed(self) -> bool:
        return self.current_state == TaskStage.PLACED

    def get_state_name(self) -> str:
        return self.current_state.name


BRIDGE = CvBridge()
# 注意：CAMERA_IMAGES 的键将在 load_config 后初始化
CAMERA_IMAGES = {} 
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
    """构造符合 OpenPI 输入格式的 obs"""
    camera_names = CFG['ros']['camera_names']
    
    missing_cams = [cam for cam in camera_names if CAMERA_IMAGES.get(cam) is None]
    if missing_cams:
        rospy.logwarn(f"Missing camera images: {missing_cams}. Skipping this observation.")
        return None

    left_arm = STATE.get("/left_arm/joint_states")
    gripper = STATE.get("/left_arm_gripper/joint_states")
    if left_arm is None or gripper is None:
        rospy.logwarn("Missing joint states. Skipping this observation.")
        return None

    images = {}
    for cam in camera_names:
        try:
            images[cam] = cv2.cvtColor(CAMERA_IMAGES[cam], cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Error converting camera {cam} image to RGB: {e}")
            return None

    joint_state = np.concatenate([np.array(left_arm).flatten(), np.array(gripper).flatten()])

    obs = {
        "state": joint_state,             
        "image": images["head"],          
        "wrist_image_left": images["left_arm"],
        "wrist_image_right": images["right_arm"],
        "prompt": CFG['inference']['task2_prompt'],
    }

    return obs


# -------------------------- Robotic Arm Control Wrapper --------------------------
def create_socket_callback(host: str, port: int, timeout: float = 5.0):
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
        
        message = {
            'timestamp': time.time(),
            'index': point_index,
            'joints': joint_data.tolist(),
            'gripper': float(gripper_data)
        }
        
        try:
            sock.send((json.dumps(message) + '\n').encode())
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            return False
    
    return socket_callback, sock


# ---------------------- Galbot 控制 ----------------------
class GalbotController:
    def __init__(self, logger):
        self.interface = GalbotControlInterface(log_level="error")
        self.publisher = ExternalDataJointPublisher(frequency=50, max_queue_size=1000)
        
        robot_cfg = CFG['robot']
        socket_callback, sock = create_socket_callback(robot_cfg['ip'], robot_cfg['port'])
        
        self.start_pulish = False
        if sock is not None:
            self.publisher.set_callback(socket_callback)
            print("Socket回调已设置")
        else:
            print("警告: Socket连接失败,将使用默认打印模式")
        self.logger = logger

    def reset_pose(self):
        self.interface.set_arm_joint_angles(
            arm_joint_angles=CFG['robot']['reset_joints'], 
            speed=1.0, 
            arm="left_arm", 
            asynchronous=True
        )
        time.sleep(2)
    
    def execute_action_realtime(self, actions: np.ndarray, is_placed_state: bool = False):
        gripper_cfg = CFG['robot']['gripper']
        offset = gripper_cfg['offset']
        max_limit = gripper_cfg['action_limit']
        threshold = gripper_cfg['threshold']

        if isinstance(actions, np.ndarray):
            if not is_placed_state:
                # PLACING status: Apply offset and clip
                actions[:, 7] = np.maximum(actions[:, 7] - offset, 0.0)
                actions[:, 7] = np.minimum(actions[:, 7], max_limit)
            else:
                # PLACED status: Lock gripper
                actions[:, 7] = threshold # 或者使用配置中的特定锁定值

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
        try:
            payload = msgpack_numpy.packb(obs, use_bin_type=True)
            self.ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
            result = self.ws.recv()

            if isinstance(result, bytes):
                result = msgpack_numpy.unpackb(result, raw=False)
            else:
                rospy.logerr(f"Unexpected result type: {type(result)}")
                return None
            return result
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
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

    task_sm = TaskStateMachine(threshold=CFG['robot']['gripper']['threshold'])
    
    step = 1
    max_steps = CFG['inference']['max_steps']
    action_horizon = CFG['inference']['action_horizon']

    while step < max_steps and not rospy.is_shutdown():
        time.sleep(0.1)
        obs = make_observation()
        if obs is None:
            time.sleep(0.05)
            continue

        # Flush WebSocket, avoid use updated actions
        try:
            policy_client.ws.settimeout(0.001)
            while True:
                try:
                    _ = policy_client.ws.recv()
                except:
                    break
        finally:
            policy_client.ws.settimeout(2)

        start_time = time.time()
        result = policy_client.infer(obs)

        if result is None or len(result.keys()) == 0:
            rospy.logwarn("Inference failed, skipping this step.")
            continue

        infer_time = (time.time() - start_time) * 1000
        rospy.loginfo(f"Inference latency: {infer_time:.1f} ms")

        actions = np.array(result["actions"])
        
        task_sm.update(actions)
        
        controller.execute_action_realtime(actions[:action_horizon], is_placed_state=task_sm.is_placed())

        time.sleep(0.5)
        rospy.loginfo(f"---- executing step {step} | State: {task_sm.get_state_name()} ----")
        step += 1


def load_config(config_path):
    if not os.path.exists(config_path):
        rospy.logerr(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            rospy.loginfo(f"Loaded configuration from {config_path}")
            return config
        except yaml.YAMLError as exc:
            rospy.logerr(f"Error parsing YAML: {exc}")
            sys.exit(1)

def main():
    rospy.init_node("openpi_ros_agent", anonymous=True)
    
    parser = argparse.ArgumentParser(description="OpenPI ROS Agent")
    parser.add_argument("-c", "--config", type=str, default="ld_depoly_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    global CFG
    CFG = load_config(args.config)

    for cam in CFG['ros']['camera_names']:
        CAMERA_IMAGES[cam] = None

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    rospy.loginfo("Starting OpenPI ROS Agent with WebSocket Model Server...")
    
    for cam in CFG['ros']['camera_names']:
        topic = f"/cam/{cam}/wrist/color/image_raw/compressed" if "arm" in cam else f"/cam/{cam}/color/image_raw/compressed"
        rospy.Subscriber(topic, CompressedImage, lambda msg, c=cam: camera_callback(msg, c), queue_size=1)
        rospy.loginfo(f"Subscribed to {topic}")

    rospy.Subscriber("/left_arm/joint_states", JointState,
                     lambda msg, s="/left_arm/joint_states": state_callback(msg, s), queue_size=1)
    rospy.Subscriber("/left_arm_gripper/joint_states", JointState,
                     lambda msg, s="/left_arm_gripper/joint_states": state_callback(msg, s), queue_size=1)

    time.sleep(1)
    
    client = WebSocketPolicyClient(CFG['inference']['ws_url_tray'])
    run_inference_loop(client)


if __name__ == "__main__":
    main()