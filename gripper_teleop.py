#!/usr/bin/env python3
"""
Alicia-D 主臂夹爪 → Gloria-M 从臂夹爪 遥操作控制

读取 Alicia-D 主臂夹爪开合值（0~1000，0=完全关闭，1000=完全张开），
线性映射到 Gloria-M 从臂夹爪的目标角度；用 MIT 力矩模式 + 软件虚拟弹簧跟随该
目标位置，并把输出扭矩限幅在 ±GRIPPER_TAU_MAX 以内。这样夹到硬物体时闭合力矩
（进而电流/功率）会在设定上限处封顶，不会无限增大导致过流保护跳闸、夹爪失能。

独立脚本，不接入 alicia_teleop_fr3.py 的机械臂控制循环，避免影响已经调好的
机械臂遥操作逻辑；后续验证稳定后再考虑是否合并。

使用步骤：
    # Step 1: 不接硬件，用 SDK 自带的 FakeCanAdapter 验证脚本逻辑
    python gripper_teleop.py --sim

    # Step 2: 供电电压确认无误、硬件都接好后，连接真实夹爪
    python gripper_teleop.py --gripper-port auto
"""

from __future__ import annotations

import argparse
import time

from alicia_d_sdk import create_robot

from gloria_m_sdk import ControlMode, GloriaGripper, GloriaSdkError, Limits, PositionRange, Variable
from gloria_m_sdk.transport import FakeCanAdapter


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


# ==========================================================================
#  可调参数（调试时直接改这里；命令行 --xxx 仍可覆盖以下默认值）
# ==========================================================================

# ---- 主臂连接配置 ----
LEADER_PORT         = ""        # Alicia-D 串口端口，留空自动查找
LEADER_GRIPPER_TYPE = "50mm"    # Alicia-D 夹爪型号

# ---- 从臂（Gloria-M）连接配置 ----
GRIPPER_PORT        = "auto"    # 串口号；'auto' 自动检测唯一可用端口
GRIPPER_BAUD        = 921_600   # 串口波特率
GRIPPER_CMD_ID      = 0x01      # 电机命令 CAN ID
GRIPPER_FB_ID       = 0x101     # 电机反馈 CAN ID

# ---- 从臂夹爪位置范围（弧度，未标定，先用 SDK demo 默认值占位）----
# TODO: 硬件到位、电压确认后，需要实际标定这两个值（手动转到全开/全合，
#       读 g.state.position 记录下来），下面是占位值，不是标定结果。
GRIPPER_OPEN_Q      = 2.5       # 从臂全开对应角度 [rad]（占位，需标定）
GRIPPER_CLOSE_Q     = 0.0       # 从臂全合对应角度 [rad]（占位，需标定）

# ---- 控制参数 ----
RATE                = 200.0     # 控制频率 (Hz)，MIT 模式官方建议 100~500Hz

# ---- 力/电流限制（MIT 力矩模式核心参数）----
# 用软件虚拟弹簧（stiffness）把主臂位置指令转换成扭矩前馈 tau，再把 tau 限幅在
# ±GRIPPER_TAU_MAX 以内下发。夹到硬物体时，位置误差会一直增大，但 tau 已经封顶，
# 电流/功率也就跟着封顶——不会再涨，也不会因为过流保护跳闸而失能。
#
# 标定方法（需要实际硬件 + 你现在用的电源功率显示）：
#   1. 从一个偏小的 GRIPPER_TAU_MAX（如 0.5）开始，运行本脚本让从臂去夹一个硬物体
#      （比如桌沿），观察电源上显示的稳态电流。
#   2. 如果电流明显小于目标 1.36A，调大 GRIPPER_TAU_MAX；如果超过，调小。
#   3. 反复几次，直到夹紧硬物体时稳态电流稳定在 ~1.36A（对应 ~32.6W）附近，
#      且不会再随你继续捏主臂夹爪而继续升高。
# 注：这里没有电机的转矩常数 Kt 和减速比，所以只能用这种"边测边调"的方式标定，
# 不能直接从 1.36A 反算出精确的 Nm 值。
GRIPPER_TAU_MAX     = 0.8       # 最大闭合/张开扭矩 [Nm]，需按上面步骤标定

GRIPPER_STIFFNESS   = 6.0       # 虚拟弹簧刚度 [Nm/rad]，越大越快达到 tau 上限（响应更快但更"硬"）
GRIPPER_KD          = 0.3       # MIT 阻尼 [Nm·s/rad]，抑制震荡，越大越柔和

DEAD_ZONE_RAW       = 5.0       # 主臂夹爪死区（0~1000 刻度），变化小于此值不下发指令

# ---- 从臂安全限制（MIT 协议编码范围；tau 的实际上限由 GRIPPER_TAU_MAX 控制，
#      这里的 LIMITS_TMAX 只需比 GRIPPER_TAU_MAX 大，作为编码分辨率范围/最后一道硬限幅）----
LIMITS_PMAX         = 3.14
LIMITS_VMAX         = 10.0
LIMITS_TMAX         = 6.0

# ==========================================================================


def _resolve_gripper_port(port: str) -> str:
    """返回 port 原值，或在 'auto' 时自动探测唯一可用串口。"""
    if port and port.lower() != "auto":
        return port
    from serial.tools import list_ports

    found = list(list_ports.comports())
    if not found:
        raise SystemExit("[port] 未找到任何串口设备，请检查串口转 CAN 适配器是否已插入")
    if len(found) > 1:
        names = ", ".join(p.device for p in found)
        raise SystemExit(f"[port] 找到多个串口设备 ({names})，请用 --gripper-port 显式指定")
    print(f"[port] 自动检测到 {found[0].device} ({found[0].description})")
    return found[0].device


def _map_leader_to_target(raw: float, args: argparse.Namespace) -> float:
    """把主臂夹爪 0~1000（0=完全关闭，1000=完全张开）线性映射到从臂目标角度。"""
    frac = max(0.0, min(1.0, raw / 1000.0))
    return args.close_q + frac * (args.open_q - args.close_q)


def _ensure_kt_value(g: GloriaGripper) -> None:
    """若电机的 KT_Value（转矩常数）寄存器为 0，MIT 模式会被固件拒绝确认
    （电流环算不出该给多少电流）。这里现场按标准 PMSM 公式 Kt=1.5*NPP*Flux
    估算一个候选值，臨时写入（不调用 save()，断电/下次上电即恢复为 0）。

    这是标定值缺失时的应急估算，不是厂家标定的准确值；后续如果厂家提供了
    准确的 KT_Value，应改为在标定流程里正式写入并 save()。
    """
    rid_kt = int(Variable.KT_Value)
    current = g.params.read(rid_kt, timeout_s=0.2)
    if current is not None and abs(current) > 1e-9:
        return  # 已经有非零值（可能厂家已标定），不覆盖

    npp = g.params.read(int(Variable.NPP), timeout_s=0.2)
    flux = g.params.read(int(Variable.Flux), timeout_s=0.2)
    if npp is None or flux is None:
        print("[WARN] 无法读取 NPP/Flux，跳过 KT_Value 估算；MIT 模式切换可能会失败")
        return

    candidate = 1.5 * float(npp) * float(flux)
    g.params.write_f32(rid_kt, candidate)
    print(f"[WARN] 检测到 KT_Value=0（未标定），已临时写入估算值 {candidate:.6f} Nm/A"
          f"（仅本次通电有效，未写入 Flash；这是应急估算，不是厂家标定值）")


def _run_calibration(g: GloriaGripper, fake: "FakeCanAdapter | None", fb_id: int,
                      limits: Limits) -> None:
    """交互式标定：把当前全闭位置设为零点（GRIPPER_CLOSE_Q=0.0），
    再实测全开位置对应的角度，供填入 GRIPPER_OPEN_Q。

    注意：set_zero() 默认只写入电机的易失性寄存器，断电即失效；
    需要额外调用 params.save() 才会写入 Flash、掉电后依然生效——
    save() 才是真正难以撤销的一步，单独确认。
    """
    print("=" * 60)
    print("    Gloria-M 从臂夹爪标定")
    print("=" * 60)

    input("请手动把夹爪转到【完全闭合】位置，就位后按 Enter 继续...")
    if fake is not None:
        fake.queue_mit_feedback(can_id=fb_id, position=0.37, velocity=0.0, torque=0.0, limits=limits)
        g.motor.refresh()
        print(f"[SIM] 当前位置（设零点前）: {g.state.position:+.3f} rad")

    g.motor.set_zero()
    if fake is not None:
        fake.queue_mit_feedback(can_id=fb_id, position=0.0, velocity=0.0, torque=0.0, limits=limits)
    g.motor.refresh()
    print(f"[INFO] 零点已设置（易失性，尚未保存）：当前位置 = {g.state.position:+.3f} rad（应接近 0）")

    confirm = input("确认这个零点正确、要保存到 Flash 掉电后依然生效吗？"
                    "此操作难以撤销 (y/n): ").strip().lower()
    if confirm != "y":
        print("[INFO] 未保存——零点仅本次通电期间有效，断电后失效，可重新标定")
    else:
        g.params.save()
        print("[INFO] 已保存到 Flash：GRIPPER_CLOSE_Q = 0.0（掉电后依然生效）")

    input("\n请手动把夹爪转到【完全张开】位置，就位后按 Enter 继续...")
    if fake is not None:
        fake.queue_mit_feedback(can_id=fb_id, position=2.5, velocity=0.0, torque=0.0, limits=limits)
    g.motor.refresh()
    open_q = g.state.position
    print(f"\n[INFO] 全开位置 = {open_q:+.3f} rad")
    print(f"[INFO] 请把这个值填入脚本顶部的 GRIPPER_OPEN_Q（当前是 {GRIPPER_OPEN_Q}）")
    print("[INFO] GRIPPER_CLOSE_Q 已经是 0.0，不需要再改")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alicia-D 主臂夹爪 → Gloria-M 从臂夹爪 遥操作控制",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 主臂
    parser.add_argument("--leader-port", type=str, default=LEADER_PORT,
                        help="Alicia-D 串口端口（留空自动查找）")
    parser.add_argument("--leader-gripper-type", type=str, default=LEADER_GRIPPER_TYPE,
                        help="Alicia-D 夹爪型号")
    # 从臂
    parser.add_argument("--gripper-port", dest="gripper_port", type=str, default=GRIPPER_PORT,
                        help="Gloria-M 串口号；'auto' 自动检测唯一可用端口")
    parser.add_argument("--gripper-baud", type=int, default=GRIPPER_BAUD, help="串口波特率")
    parser.add_argument("--cmd-id", type=lambda s: int(s, 0), default=GRIPPER_CMD_ID,
                        help="电机命令 CAN ID")
    parser.add_argument("--fb-id", type=lambda s: int(s, 0), default=GRIPPER_FB_ID,
                        help="电机反馈 CAN ID")
    parser.add_argument("--open-q", type=float, default=GRIPPER_OPEN_Q,
                        help="从臂全开对应角度 [rad]（未标定，需实测）")
    parser.add_argument("--close-q", type=float, default=GRIPPER_CLOSE_Q,
                        help="从臂全合对应角度 [rad]（未标定，需实测）")
    parser.add_argument("--rate", type=float, default=RATE, help="控制频率 (Hz)")
    parser.add_argument("--tau-max", type=float, default=GRIPPER_TAU_MAX,
                        help="最大闭合/张开扭矩 [Nm]，需按脚本顶部注释的步骤标定")
    parser.add_argument("--stiffness", type=float, default=GRIPPER_STIFFNESS,
                        help="虚拟弹簧刚度 [Nm/rad]")
    parser.add_argument("--kd", type=float, default=GRIPPER_KD,
                        help="MIT 阻尼 [Nm·s/rad]")
    parser.add_argument("--dead-zone", type=float, default=DEAD_ZONE_RAW,
                        help="主臂夹爪死区（0~1000 刻度）")
    parser.add_argument("--sim", action="store_true",
                        help="干跑模式：用 FakeCanAdapter 模拟从臂夹爪，不连接真实硬件")
    parser.add_argument("--calibrate", action="store_true",
                        help="标定模式：交互式设置从臂零点(全闭)和全开角度，不连接主臂、不进入遥操作循环")
    args = parser.parse_args()

    print("=" * 60)
    print("    Alicia-D 主臂夹爪 → Gloria-M 从臂夹爪 遥操作")
    print("=" * 60)
    print(f"  控制频率:    {args.rate} Hz")
    print(f"  从臂位置范围: 全合={args.close_q:.2f}rad  全开={args.open_q:.2f}rad（占位值，未标定）")
    print(f"  力矩限幅:    tau_max=±{args.tau_max:.2f}Nm  stiffness={args.stiffness:.1f}Nm/rad  kd={args.kd:.2f}"
          "（需按脚本顶部注释标定 tau_max，使稳态电流≈目标值）")
    print(f"  模式:        {'标定（--calibrate）' if args.calibrate else ('仿真（--sim，无需硬件）' if args.sim else '真实硬件')}")
    print("-" * 60)

    leader = None
    if not args.calibrate:
        # ---- 连接主臂（读取夹爪开合值）；标定模式不需要主臂 ----
        print("[INFO] 连接 Alicia-D 主臂...")
        leader = create_robot(port=args.leader_port, gripper_type=args.leader_gripper_type)
        if not leader.is_connected():
            raise RuntimeError("Alicia-D 主臂连接失败")
        print("[INFO] Alicia-D 主臂已连接")

    # ---- 连接从臂夹爪 ----
    limits = Limits(pmax=LIMITS_PMAX, vmax=LIMITS_VMAX, tmax=LIMITS_TMAX)
    safe_q = PositionRange(min=min(args.open_q, args.close_q), max=max(args.open_q, args.close_q))

    fake: FakeCanAdapter | None = None
    if args.sim:
        fake = FakeCanAdapter()
        gripper_port = "unused"
        transport_kwargs = {"_transport": fake}
        print("[INFO] 仿真模式：使用 FakeCanAdapter，不会打开真实串口")
    else:
        gripper_port = _resolve_gripper_port(args.gripper_port)
        transport_kwargs = {}

    try:
        with GloriaGripper(
            gripper_port,
            baudrate=args.gripper_baud,
            command_id=args.cmd_id,
            feedback_id=args.fb_id,
            limits=limits,
            safe_position=safe_q,
            **transport_kwargs,
        ) as g:
            if fake is not None:
                # FakeCanAdapter 不会自动模拟电机应答，set_mode 内部会阻塞等待
                # CTRL_MODE 回显确认，这里手动喂一条模拟应答（rid=10 对应 CTRL_MODE）。
                fake.queue_param_reply(can_id=args.fb_id, rid=10,
                                       value=int(ControlMode.MIT), is_u32=True)
            else:
                # 真实硬件：KT_Value=0 会导致固件拒绝确认 MIT 模式，先检查/临时修补。
                _ensure_kt_value(g)
            g.motor.set_mode(ControlMode.MIT)
            print("[INFO] Gloria-M 从臂夹爪：MIT 力矩模式已开启")

            if args.calibrate:
                # 标定阶段不上使能：enable() 会让电机主动闭环保持位置、
                # 对抗外力，导致没法手动把夹爪掰到全开/全合位置。
                # set_zero()/refresh() 本身不依赖使能状态，可以直接用。
                print("[INFO] 标定模式：不上使能，夹爪可自由手动转动")
                _run_calibration(g, fake, args.fb_id, limits)
                return

            g.motor.enable()
            print("[INFO] Gloria-M 从臂夹爪：已上使能")
            if fake is not None:
                fake.queue_mit_feedback(can_id=args.fb_id, position=args.close_q,
                                        velocity=0.0, torque=0.0, limits=limits)
            g.motor.refresh()
            print(f"[INFO] 从臂初始位置: {g.state.position:.3f} rad")

            interval = 1.0 / args.rate
            last_raw = None

            print("[INFO] 开始遥操作，按 Ctrl+C 安全退出")
            try:
                while True:
                    t0 = time.perf_counter()

                    try:
                        state = leader.get_robot_state("joint_gripper", timeout=0.1)
                    except Exception:
                        continue
                    if state is None:
                        continue

                    raw = float(state.gripper)
                    if last_raw is not None and abs(raw - last_raw) < args.dead_zone:
                        raw = last_raw
                    else:
                        last_raw = raw

                    target = _map_leader_to_target(raw, args)

                    # 虚拟弹簧：位置误差 × stiffness 得到期望扭矩，再限幅在 ±tau_max。
                    # 自由空间里表现接近位置跟随；夹到硬物体后误差持续增大，
                    # 但 tau 已经封顶——闭合力矩（进而电流/功率）不会再涨。
                    error = target - g.state.position
                    tau_cmd = _clamp(args.stiffness * error, -args.tau_max, args.tau_max)

                    if fake is not None:
                        # 仿真只验证映射/限幅逻辑，不模拟真实接触力，
                        # 这里让位置"瞬间"跟到目标，方便观察 target 是否正确。
                        fake.queue_mit_feedback(can_id=args.fb_id, position=target,
                                                velocity=0.0, torque=tau_cmd, limits=limits)
                    g.motion.send_mit(kp=0.0, kd=args.kd, q=0.0, dq=0.0, tau=tau_cmd, poll=True)

                    if fake is not None:
                        print(f"\r[SIM] 主臂夹爪={raw:6.1f}  目标={target:+.3f}rad  "
                              f"从臂反馈={g.state.position:+.3f}rad  tau={tau_cmd:+.2f}Nm", end="", flush=True)

                    elapsed = time.perf_counter() - t0
                    time.sleep(max(0.0, interval - elapsed))

            except KeyboardInterrupt:
                print("\n[STOP] 收到中断信号，正在安全停止...")
            finally:
                g.motor.disable()
                print("[INFO] Gloria-M 从臂夹爪：已失能")

    except GloriaSdkError as e:
        print(f"[ERROR] Gloria-M SDK 错误: {e}")
    finally:
        if leader is not None:
            leader.disconnect()
            print("[INFO] Alicia-D 主臂已断开，程序退出")
        else:
            print("[INFO] 程序退出")


if __name__ == "__main__":
    main()
