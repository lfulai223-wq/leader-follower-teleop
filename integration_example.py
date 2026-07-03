#!/usr/bin/env python3
"""Example: consume live leader-arm data from another local process/module.

Demonstrates the two integration patterns exposed by LeaderArmStream:
1. Callback - notified synchronously, inline on the polling thread.
2. Queue    - pulled from your own thread/loop at your own pace.

This is the actual integration point for other local systems: import
LeaderArmStream into your own code (as done below) instead of running
this file directly.

Usage:
    python integration_example.py
"""

import argparse
import math
import queue
import threading
import time

from record import LeaderArmStream


def on_sample(sample):
    """Callback pattern: runs on the stream's polling thread."""
    deg = [round(a * 180.0 / math.pi, 1) for a in sample.angles]
    print(f"[callback] t={sample.timestamp:.3f} joints(deg)={deg} gripper={sample.gripper:.0f}")


def consume_queue(q, stop_event):
    """Queue pattern: runs in its own thread/loop, decoupled from sampling rate."""
    while not stop_event.is_set():
        try:
            sample = q.get(timeout=0.5)
        except queue.Empty:
            continue
        print(f"[queue]    t={sample.timestamp:.3f} gripper={sample.gripper:.0f} status={sample.run_status_text}")


def main(args):
    stream = LeaderArmStream(port=args.port, gripper_type=args.gripper_type, hz=args.hz)

    # Pattern 1: callback
    stream.add_callback(on_sample)

    # Pattern 2: queue, consumed independently by another piece of code
    q = stream.subscribe(maxsize=200)
    stop_event = threading.Event()
    consumer_thread = threading.Thread(target=consume_queue, args=(q, stop_event), daemon=True)
    consumer_thread.start()

    stream.start()
    print(f"运行中，按 Ctrl+C 停止（目标 {args.hz} Hz）...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        stream.stop()
        consumer_thread.join(timeout=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leader arm stream integration example")
    parser.add_argument("--port", type=str, default="", help="串口端口（留空自动查找）")
    parser.add_argument("--gripper_type", type=str, default="50mm", help="夹爪型号")
    parser.add_argument("--hz", type=float, default=20.0, help="目标采样频率 (Hz)")
    main(parser.parse_args())
