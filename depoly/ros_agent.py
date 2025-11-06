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
from galbot_control_interface import GalbotControlInterface


# ---------------------- CONFIG ----------------------
CAMERA_NAMES = ["head", "left_arm", "right_arm"]
CAMERA_SIZE = (321, 240)
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
    """构造符合 OpenPI 输入格式的 obs，保持 numpy.ndarray 类型"""

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
        "wrist_image": images["left_arm"],# numpy.ndarray
        "prompt": "pick up the object and lift it up."
    }

    return obs



# ---------------------- Galbot 控制 ----------------------
class GalbotController:
    def __init__(self):
        self.interface = GalbotControlInterface(log_level="error")

    def reset_pose(self):
        self.interface.set_arm_joint_angles(
            arm_joint_angles=LEFT_ARM_RESET, speed=1.0,
            arm="left_arm", asynchronous=True
        )
        time.sleep(2)

    def open_gripper(self):
        self.interface.set_gripper_status(
            width_percent=GRIPPER_OPEN, speed=1.0, force=10,
            gripper="left_gripper"
        )

    def execute_action(self, action: np.ndarray):
        arm_joints = action[:7].tolist()
        self.interface.set_arm_joint_angles(
            arm_joint_angles=arm_joints, speed=1.0,
            arm="left_arm", asynchronous=True
        )
        gripper_pos = np.clip((action[7]-0.002)/0.058, 0.0, 0.65)
        self.interface.set_gripper_status(
            width_percent=gripper_pos, speed=1.0, force=11,
            gripper="left_gripper"
        )


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


import csv
CSV_FILE = "./infer_ms.csv"
# 确保文件存在，如果不存在就写入表头
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["before_optimized_infer_ms"])
# ---------------------- 主推理循环 ----------------------
def run_inference_loop(policy_client):
    controller = GalbotController()
    action_records = []
    step = 1

    while step < DEPLOY_CONFIG["max_steps"] and not rospy.is_shutdown():
        obs = make_observation()
        if obs is None:
            time.sleep(0.05)
            continue

        progress = step / DEPLOY_CONFIG["max_steps"]
        STEP_ACTION_HORIZON = int(
            STEP_ACTION_HORIZON_MIN +
            (STEP_ACTION_HORIZON_MAX - STEP_ACTION_HORIZON_MIN) * math.exp(-DECAY_RATE * progress)
        )
        STEP_ACTION_HORIZON = max(STEP_ACTION_HORIZON_MIN, min(STEP_ACTION_HORIZON, STEP_ACTION_HORIZON_MAX))
        rospy.loginfo(f"Step {step}: horizon={STEP_ACTION_HORIZON}")

        start_time = time.time()
        result = policy_client.infer(obs)
        if result is None or len(result.keys()) == 0:
            rospy.logwarn("Inference failed, skipping this step.")
            continue
        infer_time = (time.time() - start_time) * 1000
        rospy.loginfo(f"Inference latency: {infer_time:.1f} ms")

        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([infer_time])

        actions = np.array(result["actions"])
        prev_action = None
        smoothing_factor = 0.4
        idx = 0

        rospy.loginfo(f"---- executing step {step} actions ----")
        for action in actions:
            if idx >= STEP_ACTION_HORIZON:
                break
            idx += 1

            if action[5] >= 0.736:
                action[5] = 0.736

            if prev_action is not None:
                for i in range(6):
                    action[i] = prev_action[i] * (1 - smoothing_factor) + action[i] * smoothing_factor

            # controller.execute_action(action)
            prev_action = action.copy()
            action_records.append(actions[:STEP_ACTION_HORIZON])
        step += 1

    # controller.open_gripper()
    # np.save(f"{DEPLOY_CONFIG['output_dir']}/action_records.npy", np.array(action_records))
    # rospy.loginfo(f"Saved actions to {DEPLOY_CONFIG['output_dir']}")


# ---------------------- 主函数 ----------------------
def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    rospy.init_node("openpi_ros_agent", anonymous=True)

    rospy.loginfo("Starting OpenPI ROS Agent with WebSocket Model Server...")
    for cam in CAMERA_NAMES:
        topic = f"/cam/{cam}/wrist/color/image_raw/compressed" if "arm" in cam else f"/cam/{cam}/color/image_raw/compressed"
        rospy.Subscriber(topic, CompressedImage, lambda msg, c=cam: camera_callback(msg, c), queue_size=2)
        rospy.loginfo(f"Subscribed to {topic}")

    rospy.Subscriber("/left_arm/joint_states", JointState,
                     lambda msg, s="/left_arm/joint_states": state_callback(msg, s), queue_size=2)
    rospy.Subscriber("/left_arm_gripper/joint_states", JointState,
                     lambda msg, s="/left_arm_gripper/joint_states": state_callback(msg, s), queue_size=2)

    time.sleep(1)
    client = WebSocketPolicyClient(DEPLOY_CONFIG["ws_url"])
    run_inference_loop(client)


if __name__ == "__main__":
    main()
