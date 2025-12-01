import os
import shutil
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torchcodec.decoders as codecs

def validate_video(video_path):
    """
    尝试解码视频所有帧。
    返回: (是否有效, 错误信息)
    """
    try:
        decoder = codecs.VideoDecoder(str(video_path))
        # 必须遍历所有帧才能确保中间没有数据损坏
        for _ in decoder:
            pass
        return True, None
    except Exception as e:
        return False, str(e)

def check_episode(episode_path):
    """
    检查一个 Episode 目录下的所有视频。
    如果发现任何一个坏视频，立即判定该 Episode 为损坏。
    返回: (episode_path, is_valid, error_message, corrupt_file_path)
    """
    # 递归查找该 episode 下的所有 MP4 文件
    video_files = list(episode_path.rglob("*.mp4"))

    if not video_files:
        return episode_path, False, "No videos found in directory", None

    for video_file in video_files:
        is_valid, err = validate_video(video_file)
        if not is_valid:
            return episode_path, False, err, video_file

    return episode_path, True, None, None

def main():
    parser = argparse.ArgumentParser(description="Scan and clean corrupt episode directories.")
    parser.add_argument(
        "--roots",
        nargs='+',
        required=True,
        help="List of root directories containing episode folders (e.g., /path/to/data1 /path/to/data2)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker processes"
    )

    args = parser.parse_args()

    # 1. 收集所有目标路径下的 episode 目录
    all_episodes = []
    print(f"[INFO] Scanning root directories...")

    for root_str in args.roots:
        root_path = Path(root_str)
        if not root_path.exists():
            print(f"[WARNING] Path does not exist, skipping: {root_path}")
            continue

        # 假设目录名以 "episode_" 开头
        episodes = [p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("episode_")]
        print(f"[INFO] Found {len(episodes)} episodes in {root_path}")
        all_episodes.extend(episodes)

    if not all_episodes:
        print("[ERROR] No episode directories found. Exiting.")
        return

    print(f"[INFO] Starting validation for {len(all_episodes)} total episodes with {args.workers} workers.")
    print("-" * 60)

    # 2. 并行检查
    corrupt_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_ep = {executor.submit(check_episode, ep): ep for ep in all_episodes}

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_ep), total=len(all_episodes), unit="ep"):
            ep_path, is_valid, msg, corrupt_file = future.result()

            if not is_valid:
                corrupt_results.append({
                    "episode_path": ep_path,
                    "reason": msg,
                    "corrupt_file": corrupt_file
                })

    # 3. 报告与用户确认
    print("-" * 60)
    if not corrupt_results:
        print("[INFO] validation complete. No corrupt episodes found.")
        return

    print(f"[RESULT] Found {len(corrupt_results)} corrupt episodes:")
    for idx, item in enumerate(corrupt_results, 1):
        ep_path = item['episode_path']
        bad_file = item['corrupt_file']
        reason = item['reason']

        print(f"\n{idx}. Episode Directory: {ep_path}")
        if bad_file:
            try:
                rel_bad_file = bad_file.relative_to(ep_path)
            except ValueError:
                rel_bad_file = bad_file
            print(f"   Corrupt Video:    {rel_bad_file}")
        print(f"   Error Detail:     {reason}")

    print("\n" + "=" * 60)
    print(f"[ACTION REQUIRED] The {len(corrupt_results)} directories listed above are identified as corrupt.")
    user_input = input("Are you sure you want to PERMANENTLY DELETE these directories? (y/n): ").strip().lower()

    if user_input == 'y':
        print(f"[INFO] Deleting {len(corrupt_results)} directories...")
        deleted_count = 0
        for item in corrupt_results:
            ep_path = item['episode_path']
            try:
                shutil.rmtree(ep_path)
                deleted_count += 1
                print(f"[DELETED] {ep_path}")
            except Exception as e:
                print(f"[FAILED] Could not delete {ep_path}: {e}")

        print("-" * 60)
        print(f"[INFO] Operation complete. Deleted {deleted_count}/{len(corrupt_results)} directories.")
    else:
        print("[INFO] Operation cancelled by user. No files were deleted.")

if __name__ == "__main__":
    main()
