#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append("/home/abc/dev/galbot_porter/scripts")

import time
import json
import logging
import argparse
import socket
import threading
import signal
from typing import Dict, Optional, Any, List

import yaml
import numpy as np
import cv2
import websocket
from pynput import keyboard
# from galbot_porter.scripts.seach_grasp import FSM

# ROS Imports
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from openpi_client import msgpack_numpy
from galbot_control_interface import GalbotControlInterface
from joint_pulisher import ExternalDataJointPublisher

# import galbot_porter for grasp using foundation_pose
import galbot_porter.scripts.main_dev as galbot_porter_main


# ==============================================================================
#                               CONFIG & CONSTANTS
# ==============================================================================

DEBUG_MODE = False

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

    @classmethod
    def banner(cls, text: str, color=BLUE):
        print(f"\n{color}" + "=" * 60)
        print(f"   {text}")
        print("=" * 60 + f"{cls.ENDC}\n")

    @classmethod
    def log_step(cls, step: int, latency: float):
        print(f"{cls.CYAN}[STEP {step:03d}]{cls.ENDC} Latency: {latency:.1f}ms")

    @classmethod
    def log_success(cls, msg: str):
        print(f"{cls.GREEN}[SUCCESS] {msg}{cls.ENDC}")

    @classmethod
    def log_error(cls, msg: str):
        print(f"{cls.FAIL}[ERROR] {msg}{cls.ENDC}")

    @classmethod
    def log_warn(cls, msg: str):
        print(f"{cls.WARNING}[WARN] {msg}{cls.ENDC}")

    @classmethod
    def ask_user(cls, prompt: str, task_name: str) -> bool:
        # if not DEBUG_MODE:
        #     print(f" [G-A-L-B-O-T] {cls.BOLD}Skip User INTERACTION due to RELEASE MODE!!!{cls.ENDC}")
        #     return True
        print(f"\n{cls.WARNING}" + "-" * 60)
        print(f" [G-A-L-B-O-T] INTERACTION REQUIRED: {cls.BOLD}{task_name}{cls.ENDC}{cls.WARNING}")
        print(f" [G-A-L-B-O-T] Instruction: {prompt}")
        print("-" * 60 + f"{cls.ENDC}")
        
        try:
            user_input = input(f">>> Press {cls.GREEN}'y'{cls.ENDC} to proceed, or any other key to abort: ").strip().lower()
            if user_input == '':
                return True
            choice = user_input[-1]
        except EOFError:
            return False

        if choice == 'y':
            print(f"{cls.GREEN}>>> Confirmed. Starting...{cls.ENDC}\n")
            return True
        else:
            print(f"{cls.FAIL}>>> Aborted by user.{cls.ENDC}\n")
            return False


# ==============================================================================
#                               SENSOR MANAGER
# ==============================================================================

class SensorManager:
    """管理 ROS 订阅和观测数据的构建"""
    
    def __init__(self, cfg: Dict):
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
        for cam in self.cfg['ros']['camera_names']:
            if "arm" in cam:
                topic = f"/cam/{cam}/wrist/color/image_raw/compressed"
            else:
                topic = f"/cam/{cam}/color/image_raw/compressed"
            
            rospy.Subscriber(
                topic, CompressedImage, 
                lambda msg, c=cam: self._camera_cb(msg, c), 
                queue_size=1
            )
            rospy.loginfo(f"Subscribed to Camera: {topic}")

        rospy.Subscriber(
            "/left_arm/joint_states", JointState, 
            lambda msg: self._state_cb(msg, "/left_arm/joint_states"), 
            queue_size=1
        )
        rospy.Subscriber(
            "/left_arm_gripper/joint_states", JointState, 
            lambda msg: self._state_cb(msg, "/left_arm_gripper/joint_states"), 
            queue_size=1
        )

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
        with self.lock:
            # Check data integrity
            missing_cams = [k for k, v in self.camera_images.items() if v is None]
            if missing_cams:
                rospy.logwarn_throttle(2, f"Missing camera images: {missing_cams}. Skipping observation.")
                return None

            left_arm = self.joint_states.get("/left_arm/joint_states")
            gripper = self.joint_states.get("/left_arm_gripper/joint_states")
            if left_arm is None or gripper is None:
                return None

            # 2. Process Images (BGR -> RGB)
            processed_imgs = {
                k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB) 
                for k, v in self.camera_images.items()
            }

            # 3. Concatenate Joint States
            joint_state_arr = np.concatenate([
                np.array(left_arm).flatten(), 
                np.array(gripper).flatten()
            ])

            return {
                "state": joint_state_arr,
                "image": processed_imgs.get("head"),
                "wrist_image_left": processed_imgs.get("left_arm"),
                "wrist_image_right": processed_imgs.get("right_arm"),
                "prompt": prompt
            }


# ==============================================================================
#                               NETWORK CLIENTS
# ==============================================================================

class WebSocketPolicyClient:
    """处理与模型服务器的 WebSocket 通信"""
    
    def __init__(self, ws_url: str):
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

    def infer(self, obs: Dict) -> Optional[Any]:
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


# ==============================================================================
#                               ROBOT CONTROLLER
# ==============================================================================

class GalbotController:
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.interface = GalbotControlInterface(log_level="error")
        self.publisher = ExternalDataJointPublisher(frequency=50, max_queue_size=1000)
        self.is_publishing = False
        self.sock = None

        config_path = os.path.join("/home/abc/dev/openpi/depoly/galbot_porter", "config/")
        self.galbot_porter_fsm = galbot_porter_main.FSM( log_level="INFO", config_path=config_path,)
        
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
        if self.sock is None: 
            return False
            
        msg = {
            'timestamp': time.time(),
            'index': point_index,
            'joints': joint_data.tolist(),
            'gripper': float(gripper_data)
        }
        try:
            self.sock.send((json.dumps(msg) + '\n').encode())
            return True
        except Exception:
            return False

    def execute_actions(self, actions: np.ndarray):
        """处理并执行动作序列"""
        gripper_cfg = self.cfg['robot']['gripper']
        offset = gripper_cfg['offset']
        max_limit = gripper_cfg['action_limit'] # Preserved comment

        if isinstance(actions, np.ndarray):
            actions[:, 7] = np.maximum(actions[:, 7] - offset, 0.0)
            actions[:, 7] = np.minimum(actions[:, 7], max_limit)

        self.publisher.add_joints(actions)

        if not self.is_publishing:
            self.publisher.start(loop=False, async_mode=True)
            self.is_publishing = True

    # --------------------------------------------------------------------------
    # Motion Primitives Wrappers
    # --------------------------------------------------------------------------

    def move_arm(self, joints, speed=0.7, arm="left_arm", async_mode=False):
        try:
            self.interface.set_arm_joint_angles(
                arm_joint_angles=joints, speed=speed, arm=arm, asynchronous=async_mode
            )
            if not async_mode: 
                time.sleep(0.5)
        except Exception as e:
            TermUI.log_error(f"Move arm ({arm}) failed: {e}")

    def move_arm_slow(self, joints, speed=0.3, arm="left_arm", async_mode=False):
        try:
            self.interface.set_arm_joint_angles(
                arm_joint_angles=joints, speed=speed, arm=arm, asynchronous=async_mode
            )
            if not async_mode: 
                time.sleep(0.5)
        except Exception as e:
            TermUI.log_error(f"Move arm ({arm}) failed: {e}")

    def move_legs(self, joints, speed=0.7, async_mode=False):
        try:
            self.interface.set_leg_joint_angles(
                leg_joint_angles=joints, speed=speed, asynchronous=async_mode
            )
        except Exception as e:
            TermUI.log_error(f"Move legs failed: {e}")

    def _set_gripper(self, side: str, width: float):
        gripper_name = f"{side}_gripper"
        try:
            self.interface.set_gripper_status(
                width_percent=width, speed=0.2, force=20, gripper=gripper_name
            )
        except Exception as e:
            TermUI.log_error(f"Gripper {side} action failed: {e}")

    def gripper_close(self, side="left"):
        self._set_gripper(side, 0.005)

    def gripper_open(self, side="left"):
        self._set_gripper(side, 0.8)

    def gripper_open_half(self, side="left"):
        self._set_gripper(side, 0.5)

    # --------------------------------------------------------------------------
    # Status Checks & Safety
    # --------------------------------------------------------------------------

    def wait_until_done(self, part: str):
        """阻塞直到指定部件运动完成"""
        hw_map = {
            "left_arm": "left arm",
            "right_arm": "right arm",
            "leg": "leg"
        }
        
        if part not in hw_map:
            TermUI.log_error(f"Unknown part for wait: {part}")
            return

        readable_name = hw_map[part]
        while True:
            is_running = self.interface.get_follow_trajectory_status(hardware=part)[0]
            if not is_running:
                break
            time.sleep(0.05)
            TermUI.log_warn(f"[G-A-L-B-O-T] Waiting for {readable_name}...")

    def wait_until_all_done(self):
        self.wait_until_done("left_arm")
        self.wait_until_done("right_arm")
        self.wait_until_done("leg")
    
    def move_to_safe_pose(self):
        """移动到预定义的安全姿态"""
        # Note: Using config from 'place_spider_on_workshop_right' as per original code
        self.wait_until_all_done()
        c = self.cfg['place_spider_on_workshop_right']
        
        if DEBUG_MODE and not TermUI.ask_user("[G-A-L-B-O-T] Move to safe pose?", "SAFETY CHECK"):
            return

        self.move_arm(c['safty_left_arm_joints'], arm="left_arm", async_mode=True)
        self.move_arm(c['safty_right_arm_joints'], arm="right_arm", async_mode=True)
        self.wait_until_all_done()
    
    def move_to_safe_wait_exechange(self):
        """移动到预定义的安全姿态并等待完成"""
        if TermUI.ask_user("[G-A-L-B-O-T] Need move to safe for exechanging?", "SAFETY CHECK"):
            self.move_to_safe_pose()
        if TermUI.ask_user("[G-A-L-B-O-T] Have you exechanged?", "SAFETY CHECK"):
            TermUI.log_success("Now we do tray next.")

    def move_to_face_spider_box(self):
        """移动到面对蜘蛛盒子的预定义姿态"""
        c = self.cfg['place_spider_on_workshop_right']
        
        self.wait_until_all_done()
        if DEBUG_MODE and not TermUI.ask_user("[G-A-L-B-O-T] Move to face spider box pose?", "SAFETY CHECK"):
            return

        self.wait_until_all_done()
        self.move_legs(c['init_leg_joints'], async_mode=True)
        self.wait_until_all_done()


# ==============================================================================
#                               TASK EXECUTOR
# ==============================================================================
class TaskExecutor:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.controller = GalbotController(cfg)
        self.sensors = SensorManager(cfg)
        self.policy_client: Optional[WebSocketPolicyClient] = None 
        self.stop_requested = False
        
        self.resume_event = threading.Event()
        self.resume_event.set()

    def _on_keypress(self, key):
        """Keyboard listener callback: q=quit, p=pause, r=resume"""
        try:
            if hasattr(key, 'char'):
                if key.char == 'q':
                    self.stop_requested = True
                    return False 
                elif key.char == 'p':
                    if self.resume_event.is_set():
                        TermUI.log_warn("\n[PAUSE] Execution paused by user. Press 'r' to resume.")
                        self.resume_event.clear() # This blocks .wait() calls
                elif key.char == 'r':
                    if not self.resume_event.is_set():
                        TermUI.log_success("\n[RESUME] Resuming execution...")
                        self.resume_event.set()   # This unblocks .wait() calls
        except AttributeError:
            pass

    def _ensure_policy_client(self, url):
        if self.policy_client is None or self.policy_client.ws_url != url:
            self.policy_client = WebSocketPolicyClient(url)

    def _perform_pre_place_maneuver(self, config_section, prompt_text="Move to initial pose?", arms_move=True, legs_move=True):
        """
        通用的前置动作
        """
        c = config_section
        self.check_pause_state()
        
        # Safety Check
        if DEBUG_MODE and not TermUI.ask_user(f"[G-A-L-B-O-T] {prompt_text}", "SAFETY CHECK 2"):
            return False

        self.controller.wait_until_all_done()
        
        # 1. Move Legs
        if legs_move:
            self.controller.move_legs(c['init_leg_joints'], async_mode=True)
            if 'init_leg_joints' in c:
                time.sleep(1.5) 

        # 2. Move Arms, skip by user setting
        if arms_move:
            self.controller.move_arm(c['init_place_left_arm_joints'], arm="left_arm", async_mode=True)
            self.controller.move_arm(c['init_place_right_arm_joints'], arm="right_arm", async_mode=True)
        
        # 3. Grippers
        self.controller.gripper_close("right")
        
        if not DEBUG_MODE or TermUI.ask_user("[G-A-L-B-O-T] Close left gripper?", "INFO"):
            self.controller.gripper_close("left")
            
        return True

    def check_pause_state(self):
        """Blocks execution if resume_event is cleared"""
        if not self.resume_event.is_set():
            TermUI.log_warn("System is PAUSED. Waiting for 'r'...")
            self.resume_event.wait()

    def run_inference_stage(self, prompt, ws_url_key, max_steps, exec_horizon):
        self._ensure_policy_client(self.cfg['inference'][ws_url_key])

        # Start Keyboard Listener
        listener = keyboard.Listener(on_press=self._on_keypress)
        listener.start()
        
        self.stop_requested = False
        print(f"Checking inference... (Press 'q' to stop)")
        
        step = 1
        try:
            while step < max_steps and not rospy.is_shutdown():
                time.sleep(0.1) # Loop throttle

                if self.stop_requested:
                    print("\n🛑 Stop signal received ('q' pressed).")
                    return True

                obs = self.sensors.get_observation(prompt)
                if obs is None:
                    continue
                
                self.policy_client.flush()

                t_start = time.time()
                result = self.policy_client.infer(obs)
                if not result:
                    continue
                
                latency = (time.time() - t_start) * 1000
                TermUI.log_step(step, latency)
                
                actions = np.array(result["actions"])
                self.controller.execute_actions(actions[:exec_horizon])

                step += 1
                # Small delay to align with execution horizon
                time.sleep(exec_horizon / 50.0 + 0.18)
                
        finally:
            listener.stop()
        
        return True
    
    # --------------------------------------------------------------------------
    # Specific Tasks
    # --------------------------------------------------------------------------

    def run_task_pick_from_box(self):
        """"Task: Pick Spider from Box"""
        TermUI.banner("TASK 0: Pick Spider from Box")
        
        if TermUI.ask_user("[G-A-L-B-O-T] PICK FROM BOX BY HAND NOW!!!", "PICK"):
            TermUI.log_warn("YOU ALREADY PICKED THE SPIDER BY HAND.")

        return self._perform_pre_place_maneuver(
            self.cfg['pick_spider_from_box'], 
            "Move to placing spider pose?",
            arms_move=False,
            legs_move=True
        )

    def run_task_place_left(self):
        """Task: Place Spider on Left Workshop"""
        TermUI.banner("TASK: Place Spider on Workshop left")
        
        # 1. Pre-motion
        self.controller.wait_until_all_done()
        if not self._perform_pre_place_maneuver(
            self.cfg['place_spider_on_workshop'], 
            "Move to initial placing pose?",
            arms_move=True,
            legs_move=False
        ):
            return True # User skipped motion but didn't fail

        # 2. Inference
        self.controller.wait_until_all_done()
        if not DEBUG_MODE or TermUI.ask_user("[G-A-L-B-O-T] Model Infer [Pick Spider to left workshop]?", "START TASK"):
            self.run_inference_stage(
                prompt=self.cfg['inference']['task1_prompt'],
                ws_url_key='ws_url_left',
                max_steps=self.cfg['inference']['max_steps'],
                exec_horizon=20
            )
        else:
            TermUI.log_warn("User aborted the task.")
        return True

    def run_task_place_right(self):
        """Place Spider on right Workshop"""
        TermUI.banner("TASK: Place Spider on Workshop right")
        
        # 1. Pre-motion
        if not self._perform_pre_place_maneuver(
            self.cfg['place_spider_on_workshop_right'], 
            "Move to initial placing pose?",
            arms_move=True,
            legs_move=False
        ):
            return True

        # 2. Wait & Infer
        self.controller.wait_until_all_done()
        self.run_inference_stage(
            prompt=self.cfg['inference']['task2_prompt'],
            ws_url_key='ws_url_right',
            max_steps=self.cfg['inference']['max_steps'],
            exec_horizon=20
        )
        return True

    def run_task_pick_hardcoded_left(self):
        """Task: Pick Spider (Hardcoded Motion) - Left"""
        if not TermUI.ask_user("[G-A-L-B-O-T] HardCode [Pick Spider from workshop]?", "START TASK"):
            return False

        TermUI.banner("TASK: Pick from Workshop (Left)")
        return True

    def run_task_pick_hardcoded_right(self):
        """Task: Pick Spider (Hardcoded Motion) - Right"""
        if not TermUI.ask_user("[G-A-L-B-O-T] HardCode [Pick Spider from workshop]?", "START TASK"):
            return False

        TermUI.banner("TASK: Pick from Workshop (Right)")
        return True

    def run_task_place_tray(self):
        """Task: Place on Tray"""
        TermUI.banner("TASK: Place on Tray")
        c = self.cfg['place_spider_on_tray']

        # Pre-motion
        self.controller.move_legs(c['init_leg_joints'], async_mode=True)
        time.sleep(1.5)
        self.controller.move_arm(c['init_place_arm_joints1'], arm="left_arm", async_mode=True)
        self.controller.move_arm(c['right_arm_obs'], arm="right_arm", async_mode=True)
        
        self.controller.wait_until_all_done()
        
        if TermUI.ask_user("[G-A-L-B-O-T] Model-Infer [Place Spider to Tray]?", "START TASK"):
            self.controller.wait_until_all_done()
            self.run_inference_stage(
                prompt=self.cfg['inference']['task3_prompt'], 
                ws_url_key='ws_url_tray', 
                max_steps=500,
                exec_horizon=30
            )
        return True

    def run_workflow(self, run_mode: str):
        TermUI.banner(f"STARTING WORKFLOW: {run_mode.upper()}")
        
        # base functions
        task_place_r = self.run_task_place_right
        task_pick_box = self.run_task_pick_from_box
        task_place_l = self.run_task_place_left
        task_pick_hard_l = self.run_task_pick_hardcoded_left
        task_pick_hard_r = self.run_task_pick_hardcoded_right
        task_tray = self.run_task_place_tray
        safe_pose = self.controller.move_to_safe_pose
        wait_exechange = self.controller.move_to_safe_wait_exechange
        move_to_face_spider_box = self.controller.move_to_face_spider_box

        # foundation_pose pick
        pick_idle = self.controller.galbot_porter_fsm.idle
        pick_goto_grasp_pos = self.controller.galbot_porter_fsm.goto_grasp_position
        pick_search_grasp = self.controller.galbot_porter_fsm.sreach_target
        pick_grasp_target = self.controller.galbot_porter_fsm.grasp_target
        pick_goto_grasp_pos_first = self.controller.galbot_porter_fsm.goto_place_position_first
        pick_goto_grasp_pos_second = self.controller.galbot_porter_fsm.goto_place_position_second

        # foundation_pose grasp_left
        grasp_left = self.controller.galbot_porter_fsm.grasp_left
        grasp_right = self.controller.galbot_porter_fsm.grasp_right

        workflow_steps = []

        if run_mode == "full":
            workflow_steps = [
                safe_pose, move_to_face_spider_box, 
                # pick first spider
                pick_idle, pick_goto_grasp_pos, pick_search_grasp, pick_grasp_target, pick_goto_grasp_pos_first,
                task_pick_box, task_place_r, 
                #pick second spider
                pick_idle, pick_goto_grasp_pos, pick_search_grasp, pick_grasp_target, pick_goto_grasp_pos_second,
                task_pick_box, task_place_l, wait_exechange, 
                # grasp left
                grasp_left,
                task_pick_hard_l,
                safe_pose, task_tray, 
                safe_pose, move_to_face_spider_box, 
                # grasp right
                grasp_right,
                task_pick_hard_r,
                safe_pose, task_tray,

                safe_pose, move_to_face_spider_box
            ]
        elif run_mode == "from_right":
            workflow_steps = [
                move_to_face_spider_box, 
                # pick first spider
                pick_idle, pick_goto_grasp_pos, pick_search_grasp, pick_grasp_target, pick_goto_grasp_pos_first,
                task_pick_box, task_place_r, 
                #pick second spider
                pick_idle, pick_goto_grasp_pos, pick_search_grasp, pick_grasp_target, pick_goto_grasp_pos_second,
                task_pick_box, task_place_l, wait_exechange, 
                # grasp left
                grasp_left,
                task_pick_hard_l,
                safe_pose, task_tray, 
                safe_pose, move_to_face_spider_box, 
                # grasp right
                grasp_right,
                task_pick_hard_r,
                safe_pose, task_tray,

                safe_pose, move_to_face_spider_box
            ]
        elif run_mode == "from_hardcoded_left":
            workflow_steps = [
                wait_exechange, 
                # grasp left
                grasp_left,
                task_pick_hard_l,
                safe_pose, task_tray, 
                safe_pose, move_to_face_spider_box, 
                # grasp right
                grasp_right,
                task_pick_hard_r,
                safe_pose, task_tray,

                safe_pose, move_to_face_spider_box
            ]
        elif run_mode == "from_hardcoded_right":
            workflow_steps = [
                safe_pose, move_to_face_spider_box, 
                # grasp right
                grasp_right,
                task_pick_hard_r,
                safe_pose, task_tray,

                safe_pose, move_to_face_spider_box
            ]
        elif run_mode == "tray":
            workflow_steps = [safe_pose, task_tray]
        elif run_mode == "only_left":
            workflow_steps = [safe_pose, task_place_l]
        elif run_mode == "only_right":
            workflow_steps = [safe_pose, task_place_r]
        elif run_mode == "only_tray":
            workflow_steps = [safe_pose, task_tray]
        elif run_mode == "place_loop":
            workflow_steps = [
                task_pick_box, task_place_r, 
                task_pick_box, task_place_l
            ]
        else:
            TermUI.log_error(f"Unknown mode: {run_mode}")
            return

        # Execute Workflow
        for step_func in workflow_steps:
            # Check return value to see if we should stop (some tasks return False on abort)
            result = step_func()
            if result is False: 
                TermUI.log_warn("Workflow interrupted by user or error.")
                return

        TermUI.banner("ALL TASKS COMPLETED SUCCESSFULLY", color=TermUI.GREEN)

def load_config(config_path):
    if not os.path.exists(config_path):
        TermUI.log_error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def signal_handler(sig, frame):
    print('\n [G-A-L-B-O-T] Interrupt received, shutting down...')
    sys.exit(0)

def main():
    rospy.init_node("galbot_pi", anonymous=True)
    
    parser = argparse.ArgumentParser(description="GalbotPI ROS Agent")
    parser.add_argument("-c", "--config", type=str, default="ld_depoly_config.yaml", help="Path to config file")
    parser.add_argument("-mode", "--mode", type=str, default="full", help="Run mode")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    
    TermUI.banner("GalbotPi ROS Agent Initializing...")
    
    try:
        executor = TaskExecutor(cfg)
        executor.run_workflow(args.mode)
    except KeyboardInterrupt:
        TermUI.banner("Stopping Agent (Ctrl+C)", color=TermUI.FAIL)
    except Exception as e:
        TermUI.log_error(f"Unexpected Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()