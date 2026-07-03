#!/usr/bin/env python3
"""主臂轨迹录制工具（单条 / 多集数据集）。

合并自：leader_arm_stream.py、record_trajectory.py、collect_dataset.py

子命令：
  single   录制一条轨迹到 CSV
  dataset  循环录制多条 episode，自动维护 dataset_index.csv

使用示例：
  python record.py single --output trajectories/demo_001.csv
  python record.py single --duration 10
  python record.py dataset --session pick_and_place
  python record.py dataset --session pick_and_place --num_episodes 20 --episode_duration 8
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import threading
import time
from dataclasses import dataclass
from queue import Full, Queue
from typing import Callable, Dict, List, Optional

from alicia_d_sdk import create_robot
from alicia_d_sdk.utils import precise_sleep


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ArmSample:
    timestamp: float
    angles: List[float]           # 6 关节角，弧度
    gripper: float                 # 0-1000（0=完全关闭，1000=完全张开）
    run_status_text: str
    pose: Optional[dict] = None   # 仅在 include_pose=True 时填充


# ---------------------------------------------------------------------------
# 主臂数据流（后台轮询 + 发布/订阅）
# ---------------------------------------------------------------------------

class LeaderArmStream:
    """后台轮询主臂并将每帧数据广播给所有订阅者（回调 / Queue）。"""

    def __init__(self, port: str = "", gripper_type: str = "50mm", hz: float = 100.0,
                 include_pose: bool = False, robot=None):
        self.hz = hz
        self.include_pose = include_pose
        self._owns_robot = robot is None
        self.robot = robot if robot is not None else create_robot(port=port, gripper_type=gripper_type)
        self._callbacks: List[Callable[[ArmSample], None]] = []
        self._queues: List[Queue] = []
        self._latest: Optional[ArmSample] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def add_callback(self, fn: Callable[[ArmSample], None]) -> None:
        self._callbacks.append(fn)

    def subscribe(self, maxsize: int = 0) -> "Queue[ArmSample]":
        q: "Queue[ArmSample]" = Queue(maxsize=maxsize)
        self._queues.append(q)
        return q

    def get_latest(self) -> Optional[ArmSample]:
        with self._lock:
            return self._latest

    def start(self) -> None:
        if not self.robot.is_connected():
            raise RuntimeError("机械臂未连接，无法开始采集")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._owns_robot and self.robot.is_connected():
            self.robot.disconnect()

    def _run(self) -> None:
        interval = 1.0 / self.hz
        spin_threshold = 0.002 if interval <= 0.010 else 0.010
        while not self._stop_event.is_set():
            start = time.perf_counter()
            state = self.robot.get_robot_state("joint_gripper", timeout=interval)
            if state is not None:
                pose = self.robot.get_pose() if self.include_pose else None
                sample = ArmSample(
                    timestamp=state.timestamp,
                    angles=list(state.angles),
                    gripper=state.gripper,
                    run_status_text=state.run_status_text,
                    pose=pose,
                )
                with self._lock:
                    self._latest = sample
                for fn in self._callbacks:
                    try:
                        fn(sample)
                    except Exception:
                        pass
                for q in self._queues:
                    try:
                        q.put_nowait(sample)
                    except Full:
                        pass
            precise_sleep(interval - (time.perf_counter() - start), spin_threshold=spin_threshold)


# ---------------------------------------------------------------------------
# CSV 工具
# ---------------------------------------------------------------------------

CSV_FIELDS = ["timestamp", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
              "gripper", "run_status"]
CSV_POSE_FIELDS = ["pos_x", "pos_y", "pos_z", "euler_x", "euler_y", "euler_z"]


def sample_to_row(sample: ArmSample, angle_format: str = "deg") -> list:
    angles = sample.angles
    if angle_format == "deg":
        angles = [a * 180.0 / math.pi for a in angles]
    row = [sample.timestamp, *angles, sample.gripper, sample.run_status_text]
    if sample.pose is not None:
        pos = sample.pose["position"]
        euler = sample.pose["euler_xyz"]
        if angle_format == "deg":
            euler = [e * 180.0 / math.pi for e in euler]
        row += [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]]
    return row


def write_samples_csv(path: str, samples: List[ArmSample], angle_format: str = "deg",
                      include_pose: bool = False, metadata: Optional[Dict] = None) -> None:
    fields = list(CSV_FIELDS)
    if include_pose:
        fields += CSV_POSE_FIELDS
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        writer = csv.writer(f)
        writer.writerow(fields)
        for sample in samples:
            writer.writerow(sample_to_row(sample, angle_format=angle_format))


# ---------------------------------------------------------------------------
# 单条录制（原 record_trajectory.py）
# ---------------------------------------------------------------------------

def cmd_single(args: argparse.Namespace) -> None:
    samples: List[ArmSample] = []
    stream = LeaderArmStream(port=args.port, gripper_type=args.gripper_type,
                             hz=args.hz, include_pose=args.include_pose)
    stream.add_callback(samples.append)
    stream.start()
    print(f"✓ 已连接，目标采样频率 {args.hz} Hz")

    try:
        if args.duration is not None:
            print(f"录制中，将在 {args.duration:.1f} 秒后自动停止...")
            time.sleep(args.duration)
        else:
            input("按 Enter 开始录制（之后可手动拖动机械臂）...")
            samples.clear()
            print("录制中，按 Enter 结束...")
            input()
    except KeyboardInterrupt:
        print("\n录制被中断")
    finally:
        stream.stop()

    print(f"✓ 录制结束，共采集 {len(samples)} 个采样点")
    if not samples:
        print("✗ 未采集到数据，未保存文件")
        return

    metadata = {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port": args.port or "auto",
        "gripper_type": args.gripper_type,
        "hz_target": args.hz,
        "angle_format": args.format,
        "sample_count": len(samples),
        "duration_s": f"{samples[-1].timestamp - samples[0].timestamp:.3f}",
    }
    write_samples_csv(args.output, samples, angle_format=args.format,
                      include_pose=args.include_pose, metadata=metadata)
    print(f"✓ 已保存到 {args.output}")


# ---------------------------------------------------------------------------
# 多集数据集录制（原 collect_dataset.py）
# ---------------------------------------------------------------------------

def _next_episode_index(session_dir: str) -> int:
    existing = [f for f in os.listdir(session_dir) if f.startswith("episode_") and f.endswith(".csv")]
    nums = []
    for f in existing:
        try:
            nums.append(int(f[len("episode_"):-len(".csv")]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def _record_one_episode(samples: list, duration: Optional[float]) -> list:
    samples.clear()
    if duration is not None:
        print(f"录制中，将在 {duration:.1f} 秒后自动停止...")
        time.sleep(duration)
    else:
        input("按 Enter 开始录制本条 episode...")
        samples.clear()
        print("录制中，按 Enter 结束本条...")
        input()
    episode_samples = list(samples)
    samples.clear()
    return episode_samples


def cmd_dataset(args: argparse.Namespace) -> None:
    session_dir = os.path.join(args.out_dir, args.session)
    os.makedirs(session_dir, exist_ok=True)
    index_path = os.path.join(session_dir, "dataset_index.csv")
    write_header = not os.path.exists(index_path)

    samples: List[ArmSample] = []
    stream = LeaderArmStream(port=args.port, gripper_type=args.gripper_type,
                             hz=args.hz, include_pose=args.include_pose)
    stream.add_callback(samples.append)
    stream.start()
    print(f"✓ 已连接，会话目录：{session_dir}")

    episode_idx = _next_episode_index(session_dir)
    saved_count = 0

    try:
        with open(index_path, "a", newline="") as index_file:
            writer = csv.writer(index_file)
            if write_header:
                writer.writerow(["episode_file", "sample_count", "duration_s", "recorded_at"])

            while args.num_episodes is None or saved_count < args.num_episodes:
                if args.num_episodes is None:
                    choice = input(
                        f"\n按 Enter 开始第 {episode_idx} 条 episode，输入 q 结束采集: "
                    ).strip().lower()
                    if choice == "q":
                        break

                episode_samples = _record_one_episode(samples, args.episode_duration)
                if not episode_samples:
                    print("✗ 本条未采集到数据，跳过")
                    continue

                episode_name = f"episode_{episode_idx:04d}.csv"
                episode_path = os.path.join(session_dir, episode_name)
                duration_s = episode_samples[-1].timestamp - episode_samples[0].timestamp
                write_samples_csv(episode_path, episode_samples, angle_format=args.format,
                                  include_pose=args.include_pose)
                writer.writerow([episode_name, len(episode_samples), f"{duration_s:.3f}",
                                 time.strftime("%Y-%m-%d %H:%M:%S")])
                index_file.flush()
                print(f"✓ 已保存 {episode_name}（{len(episode_samples)} 点, {duration_s:.1f}s）")
                episode_idx += 1
                saved_count += 1

    except KeyboardInterrupt:
        print("\n采集会话被中断")
    finally:
        stream.stop()

    print(f"✓ 采集结束，共保存 {saved_count} 条 episode，索引文件：{index_path}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="主臂轨迹录制工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--port", type=str, default="", help="串口端口（留空自动查找）")
    shared.add_argument("--gripper_type", type=str, default="50mm", help="夹爪型号")
    shared.add_argument("--hz", type=float, default=100.0, help="目标采样频率 (Hz)")
    shared.add_argument("--format", type=str, default="deg", choices=["rad", "deg"], help="角度单位")
    shared.add_argument("--include_pose", action="store_true", help="同时记录末端姿态")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # single 子命令
    p_single = sub.add_parser("single", parents=[shared], help="录制一条轨迹到 CSV")
    p_single.add_argument("--duration", type=float, default=None,
                          help="录制时长（秒），不指定则手动按 Enter 开始/结束")
    p_single.add_argument("--output", type=str,
                          default=f"trajectories/trajectory_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                          help="输出 CSV 路径")

    # dataset 子命令
    p_dataset = sub.add_parser("dataset", parents=[shared], help="循环录制多条 episode")
    p_dataset.add_argument("--session", type=str, required=True, help="数据集会话名称（用作子文件夹名）")
    p_dataset.add_argument("--out_dir", type=str, default="datasets", help="数据集根目录")
    p_dataset.add_argument("--num_episodes", type=int, default=None,
                           help="自动采集的 episode 数量，不指定则交互式逐条确认，输入 q 结束")
    p_dataset.add_argument("--episode_duration", type=float, default=None,
                           help="每条 episode 的录制时长（秒），不指定则手动按 Enter 开始/结束")

    args = parser.parse_args()
    if args.cmd == "single":
        cmd_single(args)
    else:
        cmd_dataset(args)


if __name__ == "__main__":
    main()
