#!/usr/bin/env python3
"""
republish_rosbag.py

把指定的 ROS bag 文件重新发布到原始 topic。
用法：
    python3 republish_rosbag.py --bag /path/to/file.bag [--rate 1.0] [--topics /cam/image /joint_states] [--loop]
"""

import rospy
import rosbag
import argparse
import time

def republish_bag(bag_path, rate=1.0, topics=None, loop=False):
    """
    读取并重新发布 rosbag。
    :param bag_path: str, bag 文件路径
    :param rate: float, 播放速率（相对于原始时间戳）
    :param topics: list[str] or None，要发布的 topic 列表（None 表示全部）
    :param loop: bool, 是否循环播放
    """
    print(f"Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path)

    # 预读取所有消息（用于循环播放时复用）
    all_messages = list(bag.read_messages(topics=topics))
    if not all_messages:
        print("No messages found in the bag.")
        bag.close()
        return

    # 准备 publisher 并记录 topic 对应的消息类型
    pubs = {}
    topic_types = {}  # 存储 {topic: message_type}
    print("Initializing publishers...")
    # 从消息中获取所有需要发布的topic类型
    for topic, msg, t in all_messages:
        msg_type = type(msg).__name__  # 获取消息类型名称
        if topic not in pubs:
            pubs[topic] = rospy.Publisher(topic, type(msg), queue_size=10)
            topic_types[topic] = msg_type
            # 打印初始化的topic和对应的类型
            print(f"  Initialized publisher for {topic} (type: {msg_type})")
        if topics and len(pubs) == len(topics):
            break  # 只初始化指定的topic

    # 打印所有要发布的topic及其类型
    print("\nTopics to republish:")
    for topic, msg_type in topic_types.items():
        print(f"  {topic} (type: {msg_type})")
    print(f"Start republishing (loop: {loop})\n")

    # 循环播放逻辑
    while not rospy.is_shutdown():
        start_time = None
        last_ros_time = None
        count = 0

        for topic, msg, t in all_messages:
            if rospy.is_shutdown():
                break
            if start_time is None:
                start_time = t.to_sec()  # 记录bag内的起始时间
                last_ros_time = time.time()  # 记录实际开始播放的时间

            # 计算需要休眠的时间，模拟原始时间间隔
            elapsed_in_bag = (t.to_sec() - start_time) / rate  # bag内流逝的时间（按速率缩放）
            elapsed_real = time.time() - last_ros_time  # 实际流逝的时间
            dt = elapsed_in_bag - elapsed_real

            if dt > 0:
                time.sleep(dt)

            pubs[topic].publish(msg)

            # 打印进度（每10条一次），包含topic和类型
            count += 1
            if count % 10 == 0:
                msg_type = topic_types[topic]
                print(f"[{count}] Published topic: {topic} (type: {msg_type})")

        if not loop or rospy.is_shutdown():
            break  # 不循环或已关闭则退出
        print("\nLoop playback: restarting from the beginning...\n")

    bag.close()
    print("\nBag playback completed.")


def main():
    parser = argparse.ArgumentParser(description="Republish ROS bag to original topics.")
    parser.add_argument("--bag", required=True, help="Path to the rosbag file.")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback speed (default: 1.0).")
    parser.add_argument("--topics", nargs="*", help="List of topics to republish (default: all).")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously (default: False).")
    args = parser.parse_args()

    rospy.init_node("rosbag_republisher", anonymous=True)
    republish_bag(
        bag_path=args.bag,
        rate=args.rate,
        topics=args.topics,
        loop=args.loop
    )

if __name__ == "__main__":
    main()