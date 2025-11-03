import os
import sys
import time
import numpy as np
import cv2
import jax
import jax.numpy as jnp
import torch
from pathlib import Path
import rospy
import einops
from sensor_msgs.msg import CompressedImage,JointState
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from typing import Sequence, Dict, Any

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

sys.path.append('/home/abc/dev/galbot_control_interface:/home/abc/dev/galbot_utils:$PYTHONPATH')  # Adjust path as needed

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

MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1026/100000"
MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1031/60000"

norm_stats = _checkpoints.load_norm_stats(MODEL_PATH)
default_prompt = "pick up the object and lift it up."

DEPLOY_CONFIG = {
    "model_path": MODEL_PATH,  # model weights
    "is_pytorch": False,                                    # model type
    "device": "cuda",                                       # pi0
    "infer_freq": 15,                                        # Inference frequency (Hz)
    "max_steps": 50,                                        # Maximum inference steps
    "output_dir": "/home/abc/galbot_records/5",               # Record output directory
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
    for cam in CAMERA_NAMES:
        if CAMERA_IMAGES[cam] is None:
            logging.error(f"No image received from camera {cam}")
    left_arm_joints = STATE["/left_arm/joint_states"]
    left_gripper = STATE["/left_arm_gripper/joint_states"]
    
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
class GalbotController:
    def __init__(self, logger):
        self.interface = GalbotControlInterface(log_level="error")
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
        gripper_pos = np.clip((action[7]-0.01)/0.05, 0.0, 0.65)
        gripper_status = self.interface.set_gripper_status(
            width_percent=gripper_pos, 
            speed=1.0, 
            force=11, 
            gripper="left_gripper"
        )
        return arm_status, gripper_status

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
    time.sleep(1)  # waiting for subscriptions to be initialized

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

        # for cam, img in obs.items():
        #     if cam in ["image", "wrist_image"]:
        #         cv2.imwrite(
        #             f"{DEPLOY_CONFIG['output_dir']}/obs_{cam}_step{step}.png",
        #             cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        #         )

        # perform inference with pi0 policy
        start_time = time.time()
        # with torch.no_grad() if DEPLOY_CONFIG["is_pytorch"] else jax.disable_jit():
        results = policy.infer(obs) 
        infer_time = (time.time() - start_time) * 1001
        logger.info(f"Step {step}: Inference time {infer_time:.2f}ms")
        logger.info(f"Step {step}: Action: {results['actions'].round(4)}")

        prev_action = None
        idx = 0
        smoothing_factor = 0.4

        for action in results["actions"]:
            if idx >= 20:
                continue
            idx += 1

            if action[5] >= 0.736:
                action[5] = 0.736

            if prev_action is not None:
                for i in range(6):
                    action[i] = prev_action[i] * (1 - smoothing_factor) + action[i] * smoothing_factor

            galbot_control.execute_action(action)

            prev_action = action.copy()
            action_records.append(action)

        # control frequency
        time.sleep(max(1, 1/DEPLOY_CONFIG["infer_freq"] - (time.time() - start_time)))
        step += 2
    
    # open gripper 
    galbot_control.interface.set_gripper_status(
        width_percent=GRIPPER_OPEN, 
        speed=1.0, 
        force=10, 
        gripper="left_gripper"
    )

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