import asyncio
import logging
import torch
import pickle
import cv2
import numpy as np
import msgpack
import websockets
from dataclasses import dataclass
import tyro

from realtime_vla.pi0_infer import Pi0Inference
from openpi_client import msgpack_numpy


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


print(*(get_config("pi0_galbot_low_mem_finetune").data.base_config.data_transforms.inputs))

DATA_CONFIG = get_config("pi0_galbot_low_mem_finetune").data.base_config

STEP_ACTION_HORIZON_MAX = 50
STEP_ACTION_HORIZON_MIN = 5
decay_rate = 3.0

MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1026/100000"
# MODEL_PATH = "/home/abc/Documents/ckpts/pi0/ld_1031/70000"


norm_stats = _checkpoints.load_norm_stats(MODEL_PATH)
default_prompt = "pick up the object and lift it up."

DEPLOY_CONFIG = {
    "model_path": MODEL_PATH,  # model weights
    "is_pytorch": False,                                    # model type
    "device": "cuda",                                       # pi0
    "infer_freq": 20,                                       # Inference frequency (Hz)
    "max_steps": 25,                                        # Maximum inference steps
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



@dataclass
class Args:
    """Arguments for the Pi0 WebSocket binary inference server."""
    checkpoint_path: str = "/home/abc/dev/openpi/depoly/realtime_vla/converted_checkpoint_10000.pkl"
    host: str = "0.0.0.0"
    port: int = 8000
    num_images: int = 2
    traj_len: int = 50

def pad_to_32(x) -> torch.Tensor:
    """
    将 numpy.ndarray 或 torch.Tensor 补齐/截断到 32 维，并放到 CUDA 上。
    """
    if isinstance(x, torch.Tensor):
        # 转 float32 再转 numpy，避免 bfloat16 出错
        x = x.to(torch.float32).detach().cpu().numpy()

    x = x.astype(np.float32).flatten()

    if x.shape[0] > 32:
        x = x[:32]
    elif x.shape[0] < 32:
        x = np.pad(x, (0, 32 - x.shape[0]), mode='constant')

    return torch.from_numpy(x).to("cuda")

import os
def save_message(message, idx):
    """
    保存 WebSocket 收到的二进制消息
    """
    filename = os.path.join("/home/abc/dev/openpi/depoly/test", f"obs_{idx:05d}.msgpack")
    with open(filename, "wb") as f:
        f.write(message)
    print(f"Saved message {idx} -> {filename}")

def load_message(idx):
    """
    读取保存的二进制消息，并还原成 obs 字典
    """
    filename = os.path.join("/home/abc/dev/openpi/depoly/test", f"obs_{idx:05d}.msgpack")
    with open(filename, "rb") as f:
        message = f.read()
    # obs = msgpack_numpy.unpackb(message, raw=False)
    return message

class Pi0BinaryInferenceServer:
    def __init__(self, ckpt_path: str, num_images: int, traj_len: int):
        logging.info(f"Loading checkpoint from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        self.infer = Pi0Inference(ckpt, num_images, traj_len)
        self.num_images = num_images
        self.traj_len = traj_len
        self._input_transform = _transforms.compose(DEPLOY_CONFIG["input_transforms"])
        self._output_transform = _transforms.compose(DEPLOY_CONFIG["output_transforms"])

        logging.info("✅ Pi0Inference model initialized successfully.")

    def run_inference(self, obs: dict) -> dict:
        import time
        t0 = time.time()

        obs = load_message(1)

        # === input transformation ===
        inputs = self._input_transform(obs)
        # print(f"inputs: {inputs}")

        # 生成 noise，如果 input_transformation 没生成
        if "noise" in inputs:
            noise = inputs["noise"]
        else:
            noise = np.random.randn(self.traj_len, 32).astype(np.float32)

        # 转 torch 并移动到 GPU
        state_t = torch.from_numpy(inputs["state"]).to("cuda", dtype=torch.bfloat16)
        noise_t = torch.from_numpy(noise).to("cuda", dtype=torch.bfloat16)
        image_resized = cv2.resize(inputs["image"], (224, 224), interpolation=cv2.INTER_LINEAR)
        wrist_resized = cv2.resize(inputs["wrist_image"], (224, 224), interpolation=cv2.INTER_LINEAR)

        image_t = torch.from_numpy(image_resized.astype(np.uint8)).to("cuda").to(torch.bfloat16)
        wrist_t = torch.from_numpy(wrist_resized.astype(np.uint8)).to("cuda").to(torch.bfloat16)
        images_t = torch.stack([image_t, wrist_t], dim=0)

        t_stack = time.time()

        # === 推理 forward ===
        actions = self.infer.forward(images_t, pad_to_32(state_t), noise_t)
        torch.cuda.synchronize()
        t_infer = time.time()

        # === output transformation ===
        state_cpu = inputs["state"]
        if isinstance(state_cpu, torch.Tensor):
            state_cpu = state_cpu.to(torch.float32).detach().cpu().numpy()

        actions_cpu = actions
        if isinstance(actions_cpu, torch.Tensor):
            actions_cpu = actions_cpu.to(torch.float32).detach().cpu().numpy()

        # 调用 output_transform
        result = self._output_transform({"state": state_cpu, "actions": actions_cpu})
        result["actions"] = result["actions"][:, :8]


        print(result["actions"])
        t_post = time.time()

        # === 返回结果 + 推理耗时 ===
        result["timing_ms"] = {
            "total": (t_post - t0) * 1000,
            "stack_input": (t_stack - t0) * 1000,
            "forward": (t_infer - t_stack) * 1000,
            "postprocess": (t_post - t_infer) * 1000,
        }
        return result

    async def handle_client(self, websocket):
        logging.info("🔌 Client connected.")
        try:
            async for message in websocket:
                if not isinstance(message, (bytes, bytearray)):
                    logging.warning("Received non-binary message, ignoring.")
                    continue
                try:
                    # === 解码 obs ===
                    obs = msgpack_numpy.unpackb(message, raw=False)

                    # === 推理 ===
                    result = self.run_inference(obs)

                    # === 打包结果并发送 ===
                    payload = msgpack_numpy.packb(result, use_bin_type=True)
                    await websocket.send(payload)

                except Exception as e:
                    logging.exception("Inference error:")
                    err = msgpack.packb({"error": str(e)}, use_bin_type=True)
                    await websocket.send(err)
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected.")

    async def start(self, host: str, port: int):
        logging.info(f"🚀 Starting Pi0 Binary WebSocket server at ws://{host}:{port}")
        async with websockets.serve(self.handle_client, host, port, max_size=50 * 1024 * 1024):
            await asyncio.Future()  # run forever



def main(args: Args):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", force=True)
    server = Pi0BinaryInferenceServer(args.checkpoint_path, args.num_images, args.traj_len)
    asyncio.run(server.start(args.host, args.port))


if __name__ == "__main__":
    main(tyro.cli(Args))
