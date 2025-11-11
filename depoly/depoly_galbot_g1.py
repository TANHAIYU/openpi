import os
import sys
import time
import numpy as np
import cv2
import jax
import jax.numpy as jnp
import torch
import json
from pathlib import Path
import rospy
from sensor_msgs.msg import CompressedImage,JointState
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import math
from typing import Sequence, Dict, Any

from joint_pulisher import ExternalDataJointPublisher
import joint_pulisher
import logging

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)

# Import OpenPI inference core classes
from openpi_client import base_policy as _base_policy
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.policies.policy import Policy
from openpi.training import checkpoints as _checkpoints

from openpi.training.config import TrainConfig, pi0_config, LeRobotGalbotDataConfig, DataConfig, get_config
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

extra_paths = [
    "/home/abc/dev/galbot_control_interface",
    "/home/abc/dev/galbot_utils"
]
for p in extra_paths:
    if p not in sys.path:
        sys.path.append(p)

from galbot_control_interface import GalbotControlInterface


CAMERA_NAMES = ["head", "left_arm", "right_arm"]
CAMERA_SIZE = (321, 240)
GRIPPER_OPEN = 0.65        # Gripper open percentage
GRIPPER_CLOSE = 0.0       # Gripper close percentage

LD_LEFT_ARM_RESET_1 = [0.699064, -1.046655, -0.424937, -1.342327, -2.205600, 0.706614, -0.022770]
LD_LEFT_ARM_RESET_2 = [0.713709, -1.040710, -0.412449, -1.365338, -2.193161, 0.666658, -0.017354]
LD_LEFT_ARM_RESET_3 = [0.764810, -1.090516, -0.544276, -1.406995, -2.179666, 0.710904, -0.098007]

LEFT_ARM_RESET = LD_LEFT_ARM_RESET_1
# RIGHT_ARM_RESET = [0.483, 0.877, 0.131, 1.934, 0.399, 0.415, 0.021]
LEFT_ARM_RESET =  [1.0093586444854736, -1.3492074012756348, -1.134690761566162, -1.9343969821929932, -0.1084338054060936, 0.2987898588180542, 0.3599584102630615]


print(get_config("pi0_galbot_low_mem_finetune"))
print(*(get_config("pi0_galbot_low_mem_finetune").data.base_config.data_transforms.inputs))

DATA_CONFIG = get_config("pi0_galbot_low_mem_finetune").data.base_config

STEP_ACTION_HORIZON_MAX = 15
STEP_ACTION_HORIZON_MIN = 5
decay_rate = 3.0

# MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1026/20000"
# MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1031/70000"
MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1106/5000"
# MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1106/10000"

norm_stats = _checkpoints.load_norm_stats(MODEL_PATH)
default_prompt = "pick up the object and lift it up."

DEPLOY_CONFIG = {
    "model_path": MODEL_PATH,  # model weights
    "is_pytorch": False,                                    # model type
    "device": "cuda",                                       # pi0
    "infer_freq": 20,                                       # Inference frequency (Hz)
    "max_steps": 200,                                        # Maximum inference steps
    "output_dir": "/home/abc/galbot_records/",             # Record output directory
    "input_transforms" : [
        *DATA_CONFIG.repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(default_prompt),
        *DATA_CONFIG.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=DATA_CONFIG.use_quantile_norm),
        *DATA_CONFIG.model_transforms.inputs,
    ],
    "output_transforms": [
        *DATA_CONFIG.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=DATA_CONFIG.use_quantile_norm),
        *DATA_CONFIG.data_transforms.outputs,
        *DATA_CONFIG.repack_transforms.outputs,
    ],
}


# -------------------------- 4. Camera Data Processing (Adapting to OpenPI Input Format) --------------------------
BRIDGE = CvBridge()
# Camera image cache (single-frame cache to reduce pi0-model memory usage)
CAMERA_IMAGES: Dict[str, np.ndarray] = {
    "head": None,
    "left_arm": None,
    "right_arm": None
}
STATE:Dict[str, np.array] = {
    "/left_arm/joint_states": None,
    "/left_arm_gripper/joint_states": None
}

def make_galbot_sample(head_cam, left_cam, state) -> dict:
    """Creates a random input example for the galbot_g1 policy."""
    init_sample = {
        "state": np.random.rand(8),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "place the spider board in the designated position.",
    }
    init_sample["image"] = head_cam
    init_sample["wrist_image"] = left_cam
    init_sample["state"] = state

    return init_sample

def camera_callback(msg: CompressedImage, cam_name: str):
    """Camera callback function, caches images by name"""
    global CAMERA_IMAGES
    try:
        # Convert compressed image to BGR (avoid unnecessary format conversion on pi0)
        img = BRIDGE.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        CAMERA_IMAGES[cam_name] = img
    except Exception as e:
        rospy.logerr(f"Camera {cam_name} callback error: {str(e)}")

def state_callback(msg: JointState, state_name: str):
    """State callback function, caches joint states by name"""
    global STATE
    try:
        STATE[state_name] = list(msg.position)
    except Exception as e:
        rospy.logerr(f"State {state_name} callback error: {str(e)}")

def get_observation() -> Dict[str, Any]:
    """Construct observation dictionary required by OpenPI (matching Policy input format)"""

    missing = [cam for cam in CAMERA_NAMES if CAMERA_IMAGES.get(cam) is None]
    if missing:
        logging.warning(f"Missing camera images: {missing}. Skipping this observation.")
        return None

    
    left_arm_joints = STATE["/left_arm/joint_states"]
    left_gripper = STATE["/left_arm_gripper/joint_states"]
    # check states
    left_arm_joints = STATE.get("/left_arm/joint_states")
    left_gripper = STATE.get("/left_arm_gripper/joint_states")
    if left_arm_joints is None or left_gripper is None:
        logging.warning("Missing joint state(s). Skipping this observation.")
        return None
    
    # 3. Collect image data (convert to RGB to match model input)
    images = {
        cam: cv2.cvtColor(CAMERA_IMAGES[cam], cv2.COLOR_BGR2RGB) 
        for cam in CAMERA_NAMES
    }

    # # save images for debugging
    # for cam, img in images.items():
    #     cv2.imwrite(f"/home/abc/galbot_records/debug_{cam}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # 4. Collect joint states (7 left arm joints + gripper state)
    joint_state = np.concatenate([
        np.array(left_arm_joints).flatten(),
        np.array(left_gripper).flatten()
    ])
    
    # 5. Construct observation dictionary (matching OpenPI's Observation format)
    return make_galbot_sample(images["head"], images["left_arm"], joint_state)


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
            actions[:, 5] = np.maximum(actions[:, 5], 0.736)  # limit wrist joint


        self.publisher.add_joints(actions)

        if not self.start_pulish:
            self.publisher.start(loop=False, async_mode=True)
            self.start_pulish = True
        

def main():
    # initialize logger and output directory
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("pi0 model deployment script start")
    os.makedirs(DEPLOY_CONFIG["output_dir"], exist_ok=True)

    # # initialize robotic arm control
    global galbot_control
    galbot_control = GalbotController(logger, )

    # initialize camera subscriptions
    # rospy.init_node("openpi_node", anonymous=True)

    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True
    )
    logger.setLevel(logging.INFO)  # 重新设置子 logger 级别

    logger.info(f"Subscribing to camera topic: /cam/head/color/image_raw/compressed")
    rospy.Subscriber(
        f"/cam/head/color/image_raw/compressed",
        CompressedImage,
        lambda msg, c="head": camera_callback(msg, c),
        queue_size=2
    )
    logger.info(f"Subscribing to camera topic: /cam/right_arm/wrist/color/image_raw/compressed")
    rospy.Subscriber(
        f"/cam/right_arm/wrist/color/image_raw/compressed",
        CompressedImage,
        lambda msg, c="right_arm": camera_callback(msg, c),
        queue_size=2
    )
    logger.info(f"Subscribing to camera topic: /cam/left_arm/wrist/color/image_raw/compressed")
    rospy.Subscriber(
        f"/cam/left_arm/wrist/color/image_raw/compressed",
        CompressedImage,
        lambda msg, c="left_arm": camera_callback(msg, c),
        queue_size=2
    )
    logger.info(f"Subscribing to state topic: /left_arm/joint_states")
    rospy.Subscriber(
        f"/left_arm/joint_states",
        JointState,
        lambda msg, c="/left_arm/joint_states": state_callback(msg, c),
        queue_size=2
    )
    logger.info(f"Subscribing to state topic: /left_arm_gripper/joint_states")
    rospy.Subscriber(
        f"/left_arm_gripper/joint_states",
        JointState,
        lambda msg, c="/left_arm_gripper/joint_states": state_callback(msg, c),
        queue_size=2
    )
    time.sleep(2)  # waiting for subscriptions to be initialized

    # load pi0 model and create policy
    logger.info("start load pi0 model from pretrained checkpoints...")
    config = _config.get_config("pi0_galbot_low_mem_finetune")
    policy = _policy_config.create_trained_policy(config, DEPLOY_CONFIG.get("model_path"))
    print("policy:", policy)
    logger.info("pi0 model and policy initialized successfully")

    # inference and control loop
    step = 1
    action_records = []

    while step < DEPLOY_CONFIG["max_steps"] and not rospy.is_shutdown():
        obs = get_observation()
        logger.info(f"Step {step}: Observation data acquired successfully")

        # 动态调整 STEP_ACTION_HORIZON：随 step 增大而减小
        progress = step / DEPLOY_CONFIG["max_steps"]
        # linear decreasing
        # STEP_ACTION_HORIZON = int(
        #     STEP_ACTION_HORIZON_MAX - (STEP_ACTION_HORIZON_MAX - STEP_ACTION_HORIZON_MIN) * progress
        # )
        # exp decreasing

        # DECAY_RATE = 10.0
        # STEP_ACTION_HORIZON = int(
        #     STEP_ACTION_HORIZON_MAX -
        #     (STEP_ACTION_HORIZON_MAX - STEP_ACTION_HORIZON_MIN) * (1 - math.exp(-DECAY_RATE * progress))
        # )
        # STEP_ACTION_HORIZON = max(STEP_ACTION_HORIZON_MIN, min(STEP_ACTION_HORIZON, STEP_ACTION_HORIZON_MAX))
        # logger.info(f"Step {step}: STEP_ACTION_HORIZON = {STEP_ACTION_HORIZON}")

        # perform inference with pi0 policy
        start_time = time.time()
        results = policy.infer(obs) 
        infer_time = (time.time() - start_time) * 1001
        logger.info(f"Step {step}: Inference time {infer_time:.2f}ms")
        if step == 1:
            galbot_control.publisher.start(loop=False, async_mode=True)
        # logger.info(f"Step {step}: Action: {results['actions'].round(4)}")

        # prev_action = None
        # idx = 0
        # smoothing_factor = 0.4

        logger.info(f"-----------------------start execution {step}'s step actions------------------------")

        galbot_control.execute_action_realtime(results["actions"][:10])
        # time.sleep(STEP_ACTION_HORIZON * 0.06 + 0.3)
        time.sleep(0.5)
        step += 1
        logger.info(f"-----------------------end execution {step}'s step actions------------------------")


    # post-processing: save records + reset robotic arm
    np.save(f"{DEPLOY_CONFIG['output_dir']}/action_records.npy", np.array(action_records))
    logger.info(f"Action records saved to {DEPLOY_CONFIG['output_dir']}")
    
    # plot action trajectory
    plt.figure(figsize=(9, 6))
    for i in range(9):
        plt.subplot(5, 2, i+1)
        plt.plot([a[i] for a in action_records], linewidth=2)
        plt.title(f"Action Dimension {i}")
    plt.tight_layout()
    plt.savefig(f"{DEPLOY_CONFIG['output_dir']}/action_trajectory.png", dpi=97)

    # final reset
    # galbot_control.reset_pose()
    logger.info("Deployment script execution completed")


if __name__ == "__main__":
    main()