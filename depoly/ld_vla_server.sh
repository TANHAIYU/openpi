#!/bin/bash

# ==============================================================================
# 1. 本地 VLA 模型配置 & 启动函数
# ==============================================================================

WORK_DIR="$HOME/dev/openpi/depoly"
CONDA_ENV="pi0_deploy"

# 定义生成本地 Terminator 执行命令的函数
get_local_cmd() {
    local env_name=$1
    local port=$2
    # 显存限制设为 0.3，防止两个模型撑爆显存
    echo "zsh -c 'source ~/.zshrc; \
          conda activate $CONDA_ENV; \
          cd $WORK_DIR; \
          echo \"[LOCAL] Starting $env_name Server on port $port...\"; \
          export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3; \
          python3 model_server.py --env $env_name --port $port; \
          exec zsh'"
}

echo ">>> 正在启动本地 VLA 模型..."

# --- 启动窗口 1: LEFT ARM ---
terminator -T "VLA: WORKSPACE_LEFT (8000)" -x $(get_local_cmd LD_WORKSPACE_LEFT 8000) &
sleep 1

# --- 启动窗口 2: TRAY (根据你的代码，这里是第二个启动的) ---
terminator -T "VLA: TRAY (9000)" -x $(get_local_cmd LD_TRAY 9000) &
sleep 1

# --- 启动窗口 3: RIGHT ARM (如果需要启动，请取消注释) ---
# terminator -T "VLA: WORKSPACE_RIGHT (7000)" -x $(get_local_cmd LD_WORKSPACE_RIGHT 7000) &

echo ">>> 本地 VLA 模型启动命令已发送。"
echo ">>> 等待 5 秒以确保本地服务就绪..."
sleep 5


# # ==============================================================================
# # 2. 远程 SSH 控制端启动逻辑
# # ==============================================================================

# # 远程主机配置
# REMOTE_IP="192.168.100.100"
# REMOTE_USER="galbot_orin_2024"
# REMOTE_PASS="galbot"
# REMOTE_DIR="/home/galbot_orin_2024/workFiles/galbot_remote_ctrl_zhoubo/galbot_remote_ctrl"
# REMOTE_CONDA="gmp"

# # 构建远程执行的命令链
# # 注意：
# # 1. source ~/.bashrc 是为了加载 conda 环境配置 (假设远程是 Bash)
# # 2. -t 参数在 ssh 中用于分配伪终端，保留彩色输出和交互能力
# REMOTE_CMD_CHAIN="cd $REMOTE_DIR; \
#                   echo '[REMOTE] Activating Conda Environment...'; \
#                   source ~/.bashrc; \
#                   conda activate $REMOTE_CONDA; \
#                   echo '[REMOTE] Starting Main Control Script...'; \
#                   python3 scripts/main.py; \
#                   exec bash"

# # 定义本地 Terminator 启动 SSH 的命令
# # 使用 sshpass 传入密码，避免手动输入
# SSH_LAUNCH_CMD="sshpass -p '$REMOTE_PASS' ssh -t $REMOTE_USER@$REMOTE_IP \"$REMOTE_CMD_CHAIN\""

# echo ">>> 正在连接远程 Orin 并启动控制脚本..."

# # 在新的 Terminator 窗口中执行 SSH
# terminator -T "REMOTE: GALBOT ORIN CONTROL" -x zsh -c "$SSH_LAUNCH_CMD; exec zsh" &

# echo ">>> 所有任务已启动完成。"