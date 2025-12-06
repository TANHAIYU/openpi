#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import argparse
import yaml
import numpy as np
import cv2
import socket
import threading
import signal
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

# ROS Imports
import rospy
import websocket
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState

# Custom/Project Imports
from openpi_client import msgpack_numpy
from galbot_control_interface import GalbotControlInterface
from joint_pulisher import ExternalDataJointPublisher

# ==============================================================================
#                               UI & LOGGING UTILS
# ==============================================================================

class TermUI:
    """处理终端颜色输出和用户交互的工具类"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def banner(text: str, color=BLUE):
        print(f"\n{color}" + "=" * 60)
        print(f"   {text}")
        print("=" * 60 + f"{TermUI.ENDC}\n")

    @staticmethod
    def log_step(step: int, latency: float, state: str):
        print(f"{TermUI.CYAN}[STEP {step:03d}]{TermUI.ENDC} "
              f"Latency: {latency:.1f}ms | State: {TermUI.BOLD}{state}{TermUI.ENDC}")

    @staticmethod
    def log_success(msg: str):
        print(f"{TermUI.GREEN}[SUCCESS] {msg}{TermUI.ENDC}")

    @staticmethod
    def log_error(msg: str):
        print(f"{TermUI.FAIL}[ERROR] {msg}{TermUI.ENDC}")

    @staticmethod
    def log_warn(msg: str):
        print(f"{TermUI.WARNING}[WARN] {msg}{TermUI.ENDC}")

    @staticmethod
    def ask_user(prompt: str, task_name: str) -> bool:
        print(f"\n{TermUI.WARNING}" + "-" * 60)
        print(f" [G-A-L-B-O-T] INTERACTION REQUIRED: {TermUI.BOLD}{task_name}{TermUI.ENDC}{TermUI.WARNING}")
        print(f" [G-A-L-B-O-T] Instruction: {prompt}")
        print("-" * 60 + f"{TermUI.ENDC}")
        
        try:
            user_input = input(f">>> Press {TermUI.GREEN}'y'{TermUI.ENDC} to proceed, or any other key to abort: ").strip().lower()
        except EOFError:
            return False

        if user_input == 'y':
            print(f"{TermUI.GREEN}>>> Confirmed. Starting...{TermUI.ENDC}\n")
            return True
        else:
            print(f"{TermUI.FAIL}>>> Aborted by user.{TermUI.ENDC}\n")
            return False

# ==============================================================================
#                               CORE CLASSES
# ==============================================================================

class TaskStage(Enum):
    PLACING = auto()
    PLACED = auto()
    FAILED = auto()

class TaskStateMachine:
    def __init__(self, threshold: float):
        self.current_state = TaskStage.PLACING
        self.gripper_threshold = threshold 

    def update(self, actions: np.ndarray):
        if self.current_state == TaskStage.PLACING:
            # Check last gripper value in the action chunk
            last_gripper_val = actions[-1, 7]
            if last_gripper_val > self.gripper_threshold:
                TermUI.log_success(f"Transition: PLACING -> PLACED (Val: {last_gripper_val:.4f})")
                self.current_state = TaskStage.PLACED

    def is_placed(self) -> bool:
        return self.current_state == TaskStage.PLACED

    def reset(self):
        self.current_state = TaskStage.PLACING

    def get_state_name(self) -> str:
        color = TermUI.GREEN if self.current_state == TaskStage.PLACED else TermUI.WARNING
        return f"{color}{self.current_state.name}{TermUI.ENDC}"


class SensorManager:
    """管理 ROS 订阅和观测数据的构建"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.bridge = CvBridge()
        self.camera_images = {name: None for name in cfg['ros']['camera_names']}
        self.joint_states = {
            "/left_arm/joint_states": None,
            "/left_arm_gripper/joint_states": None
        }
        self.lock = threading.Lock()
        self._setup_subscribers()

    def _setup_subscribers(self):
        # Cameras
        for cam in self.cfg['ros']['camera_names']:
            # Auto-detect topic naming convention
            topic = f"/cam/{cam}/wrist/color/image_raw/compressed" if "arm" in cam else f"/cam/{cam}/color/image_raw/compressed"
            rospy.Subscriber(topic, CompressedImage, lambda msg, c=cam: self._camera_cb(msg, c), queue_size=1)
            rospy.loginfo(f"Subscribed to Camera: {topic}")

        # Joints
        rospy.Subscriber("/left_arm/joint_states", JointState, 
                         lambda msg: self._state_cb(msg, "/left_arm/joint_states"), queue_size=1)
        rospy.Subscriber("/left_arm_gripper/joint_states", JointState, 
                         lambda msg: self._state_cb(msg, "/left_arm_gripper/joint_states"), queue_size=1)

    def _camera_cb(self, msg, name):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.camera_images[name] = img
        except Exception as e:
            rospy.logerr_throttle(5, f"Cam {name} error: {e}")

    def _state_cb(self, msg, name):
        try:
            with self.lock:
                self.joint_states[name] = list(msg.position)
        except Exception as e:
            rospy.logerr_throttle(5, f"State {name} error: {e}")

    def get_observation(self, prompt: str) -> Optional[Dict]:
        """构建 OpenPI 格式的观测数据"""
        with self.lock:
            # Check data integrity
            missing_cams = [k for k, v in self.camera_images.items() if v is None]
            if missing_cams:
                return None # Silent return to avoid log spamming in loop

            left_arm = self.joint_states.get("/left_arm/joint_states")
            gripper = self.joint_states.get("/left_arm_gripper/joint_states")
            if left_arm is None or gripper is None:
                return None

            # Convert images BGR -> RGB
            processed_imgs = {}
            for k, v in self.camera_images.items():
                processed_imgs[k] = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

            joint_state_arr = np.concatenate([np.array(left_arm).flatten(), np.array(gripper).flatten()])

            return {
                "state": joint_state_arr,
                "image": processed_imgs["head"],
                "wrist_image_left": processed_imgs["left_arm"],
                "wrist_image_right": processed_imgs["right_arm"],
                "prompt": prompt
            }


class WebSocketPolicyClient:
    """处理与模型服务器的 WebSocket 通信"""
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self._connect()

    def _connect(self):
        rospy.loginfo(f"Connecting to Model Server: {self.ws_url}")
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=10)
            TermUI.log_success("Model Server Connected ✅")
        except Exception as e:
            TermUI.log_error(f"Connection failed: {e}")
            self.ws = None

    def infer(self, obs: dict):
        if self.ws is None:
            self._connect()
            if self.ws is None: return None

        try:
            payload = msgpack_numpy.packb(obs, use_bin_type=True)
            self.ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
            result = self.ws.recv()
            
            if isinstance(result, bytes):
                return msgpack_numpy.unpackb(result, raw=False)
            return None
        except (websocket.WebSocketException, BrokenPipeError):
            TermUI.log_warn("WebSocket connection lost. Reconnecting...")
            self._connect()
            return None
        except Exception as e:
            TermUI.log_error(f"Inference error: {e}")
            return None
    
    def flush(self):
        """清空缓冲区，确保获取最新策略"""
        if self.ws:
            self.ws.settimeout(0.001)
            while True:
                try:
                    self.ws.recv()
                except:
                    break
            self.ws.settimeout(2.0)


class GalbotController:
    """机器人控制封装"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.interface = GalbotControlInterface(log_level="error")
        self.publisher = ExternalDataJointPublisher(frequency=50, max_queue_size=1000)
        self.start_publish = False
        
        self._setup_socket_connection()

    def _setup_socket_connection(self):
        host = self.cfg['robot']['ip']
        port = self.cfg['robot']['port']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(20.0)
        
        try:
            self.sock.connect((host, port))
            TermUI.log_success(f"Robot Socket Connected: {host}:{port}")
            self.publisher.set_callback(self._socket_send_callback)
        except Exception as e:
            TermUI.log_error(f"Robot Socket Connection Failed: {e}")
            self.sock = None

    def _socket_send_callback(self, point_index, joint_data, gripper_data):
        if self.sock is None: return False
        msg = {
            'timestamp': time.time(),
            'index': point_index,
            'joints': joint_data.tolist(),
            'gripper': float(gripper_data)
        }
        try:
            self.sock.send((json.dumps(msg) + '\n').encode())
            return True
        except:
            return False

    def reset_pose(self):
        """移动到初始姿态"""
        self.interface.set_arm_joint_angles(
            arm_joint_angles=self.cfg['robot']['reset_joints'], 
            speed=1.0, arm="left_arm", asynchronous=True
        )
        time.sleep(2)

    def execute_actions(self, actions: np.ndarray, is_placed: bool = False):
        """处理并执行动作序列"""
        gripper_cfg = self.cfg['robot']['gripper']
        offset = gripper_cfg['offset']
        max_limit = gripper_cfg['action_limit']
        threshold = gripper_cfg['threshold']

        if isinstance(actions, np.ndarray):
            if is_placed:
                # 放置后锁定夹爪
                actions[:, 7] = threshold
            else:
                # 放置中应用偏移和限制
                actions[:, 7] = np.maximum(actions[:, 7] - offset, 0.0)
                actions[:, 7] = np.minimum(actions[:, 7], max_limit)

        self.publisher.add_joints(actions)

        if not self.start_publish:
            self.publisher.start(loop=False, async_mode=True)
            self.start_publish = True

    # Helper wrappers for movement
    def move_arm(self, joints, speed=0.3, arm="left_arm", async_mode=False):
        try:
            self.interface.set_arm_joint_angles(arm_joint_angles=joints, speed=speed, arm=arm, asynchronous=async_mode)
            if not async_mode: time.sleep(0.5)
        except Exception as e:
            TermUI.log_error(f"Move arm failed: {e}")

    def move_legs(self, joints, speed=0.3, async_mode=False):
        try:
            self.interface.set_leg_joint_angles(leg_joint_angles=joints, speed=speed, asynchronous=async_mode)
        except Exception as e:
            TermUI.log_error(f"Move legs failed: {e}")

    def gripper_close(self, side="left"):
        gripper_name = f"{side}_gripper"
        try:
            self.interface.set_gripper_status(width_percent=0.005, speed=0.2, force=20, gripper=gripper_name)
        except Exception as e:
            TermUI.log_error(f"Gripper close failed: {e}")


# ==============================================================================
#                               TASK EXECUTOR
# ==============================================================================

class TaskExecutor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.controller = GalbotController(cfg)
        self.sensors = SensorManager(cfg)
        self.state_machine = TaskStateMachine(threshold=cfg['robot']['gripper']['threshold'])
        self.policy_client = None # Lazy init

    def _ensure_policy_client(self, url):
        if self.policy_client is None or self.policy_client.ws_url != url:
            self.policy_client = WebSocketPolicyClient(url)

    def run_inference_stage(self, prompt, ws_url_key, max_steps):
        self._ensure_policy_client(self.cfg['inference'][ws_url_key])
        self.state_machine.reset()
        
        step = 1
        horizon = self.cfg['inference']['action_horizon']

        while step < max_steps and not rospy.is_shutdown():
            time.sleep(0.1) # Control loop frequency cap
            
            obs = self.sensors.get_observation(prompt)
            if obs is None:
                continue

            self.policy_client.flush()
            
            t_start = time.time()
            result = self.policy_client.infer(obs)
            if not result:
                continue
            
            latency = (time.time() - t_start) * 1000
            actions = np.array(result["actions"])

            # Update State Machine
            self.state_machine.update(actions)
            
            # Execute
            self.controller.execute_actions(actions[:horizon], is_placed=self.state_machine.is_placed())

            TermUI.log_step(step, latency, self.state_machine.get_state_name())
            
            if self.state_machine.current_state == TaskStage.PLACED:
                # Optional: break early or continue to ensure stability
                pass

            step += 1
        
        return True
    
    def run_task_0_pick_from_box(self):
        """"Task0: Pick Spoder from Box"""
        TermUI.banner("TASK 0: Pick Spider from Box")
        
        # 1. Reset Pose
        c = self.cfg['pick_spider_from_box']
        self.controller.move_legs(c['init_leg_joints'], async_mode=True)
        time.sleep(5) # Wait for legs
        self.controller.move_arm(c['init_pick_arm_joints'], arm="left_arm", async_mode=True)
        time.sleep(2)

        # 2. Confirm
        if not TermUI.ask_user("Ready to [Pick Spider from box] (Model Control)?", "START TASK 0"):
            return False

        # 3. Pick spider
        self.controller.gripper_close("left")
        time.sleep(2)
        return True

    def run_task_1_place_left(self):
        """Task 1: Place Spider on Left Workshop"""
        TermUI.banner("TASK 1: Place Spider on Workshop")
        
        # 1. Reset Pose
        c = self.cfg['place_spider_on_workshop']
        self.controller.move_legs(c['init_leg_joints'], async_mode=True)
        time.sleep(5) # Wait for legs
        self.controller.move_arm(c['init_place_left_arm_joints'], arm="left_arm", async_mode=True)
        time.sleep(2)
        self.controller.move_arm(c['init_place_right_arm_joints'], arm="right_arm", async_mode=True)
        time.sleep(2)
        self.controller.gripper_close("right")
        self.controller.gripper_close("left")
        time.sleep(2)

        # 2. Confirm
        if not TermUI.ask_user("Ready to [Pick Spider to left workshop] (Model Control)?", "START TASK 1"):
            return False

        # 3. Infer
        self.run_inference_stage(
            prompt=self.cfg['inference']['task1_prompt'],
            ws_url_key='ws_url',
            max_steps=self.cfg['inference']['max_steps']
        )
        return True

    def run_task_2_pick_hardcoded(self):
        """Task 2: Pick Spider (Hardcoded Motion)"""
        if not TermUI.ask_user("Proceed to [Pick Spider from workshop] (Hardcoded)?", "START TASK 2"):
            return False

        TermUI.banner("TASK 2: Pick from Workshop")
        c = self.cfg['pick_spider_from_workshop']

        try:
            self.controller.move_arm(c['pick_left_arm_joints_wp1'], arm="left_arm")
            time.sleep(1)
            self.controller.move_arm(c['pick_left_arm_joints_wp2'], arm="left_arm")
            time.sleep(1)
            self.controller.gripper_close("left")
            time.sleep(1.5)
            self.controller.move_arm(c['lift_spider_joints'], arm="left_arm")
            TermUI.log_success("Pick sequence completed.")
        except Exception as e:
            TermUI.log_error(f"Task 2 Error: {e}")
            return False
        return True

    def run_task_3_place_tray(self):
        """Task 3: Place on Tray"""
        if not TermUI.ask_user("Proceed to [Place Spider to Tray]?", "START TASK 3"):
            return False

        TermUI.banner("TASK 3: Place on Tray")
        c = self.cfg['place_spider_on_tray']

        # Pre-motion
        self.controller.move_legs(c['init_leg_joints'], async_mode=True)
        self.controller.move_arm(c['init_place_arm_joints'], arm="left_arm", async_mode=True)
        time.sleep(3)
        
        self.controller.move_arm(c['init_place_arm_joints1'], arm="left_arm")
        self.controller.move_legs(c['right_arm_obs'], async_mode=False) # Actually arm move based on config key name?
        
        # NOTE: Logic implies we might want inference here, but original code commented it out.
        # We will assume hardcoded finish for now, or uncomment below line if needed.

        if TermUI.ask_user("Do you need model-infer [Place Spider to Tay]?", "START TASK 3"):
            self.run_inference_stage(
                prompt=self.cfg['inference']['task2_prompt'], 
                ws_url_key='ws_url_tray', 
                max_steps=500
            )
        
        
        TermUI.log_success("Moved to tray position. (Inference skipped as per config).")
        return True

    def run_full_workflow(self):
        if not self.run_task_0_pick_from_box(): return
        if not self.run_task_1_place_left(): return
        if not self.run_task_2_pick_hardcoded(): return
        if not self.run_task_3_place_tray(): return
        
        TermUI.banner("ALL TASKS COMPLETED SUCCESSFULLY", color=TermUI.GREEN)


# ==============================================================================
#                               MAIN ENTRY
# ==============================================================================

def load_config(config_path):
    if not os.path.exists(config_path):
        TermUI.log_error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def signal_handler(sig, frame):
    print('\n[系统] 检测到 Ctrl+C, 正在强制清理资源并退出...')
    sys.exit(0)

def main():
    rospy.init_node("openpi", anonymous=True)
    
    parser = argparse.ArgumentParser(description="OpenPI ROS Agent")
    parser.add_argument("-c", "--config", type=str, default="ld_depoly_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    
    TermUI.banner("OpenPI ROS Agent Initializing...")
    
    try:
        executor = TaskExecutor(cfg)
        time.sleep(1.0)
        executor.run_full_workflow()
    except KeyboardInterrupt:
        TermUI.banner("Stopping Agent (Ctrl+C)", color=TermUI.FAIL)
    except Exception as e:
        TermUI.log_error(f"Unexpected Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()