"""ServoCart 平滑轨迹下发 SDK。

本模块把原先 `servo.py` 中的“读取点位 -> 轨迹规划 -> ServoCart 下发”流程封装为可复用调用。

用法示例：

    from servo_cart_sdk import ServoCartConfig, run_servo_cart

    cfg = ServoCartConfig(points=[[x, y, z, rx, ry, rz], ...])
    stats = run_servo_cart(cfg)
    print(stats)

说明：
- 本 SDK 只负责“点位 list -> 轨迹规划 -> ServoCart 下发”。不做读文件，不做回 HOME。
- 回 HOME / 去 HOME 等业务动作请放在你的运行入口（例如 `servo.py`）。
- 默认参数严格继承自你当前仓库根目录 `servo.py` 的常量区。
- SDK 本身不依赖 GUI；只要 Fairino Python SDK 可用即可实机运行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence
from spline_own import TrajectoryPlan, build_trajectory_plan, shrink_points_to_chord

try:
    from fairino390.linux.fairino import Robot
except Exception:  # pragma: no cover
    Robot = None


@dataclass
class ServoCartConfig:
    """ServoCart 下发配置（默认值来自原 `servo.py`）。

    字段按模块分组：

    连接与坐标系
    - robot_ip: 控制器 IP，用于 `Robot.RPC(robot_ip)`。
    - tool/user: 工具与用户坐标系编号，传给 MoveCart/MoveJ。

    点位输入
    - points: 点位列表（每个点至少 6 个元素 [x,y,z,rx,ry,rz]）。
    - swap_rpy_order: 姿态三轴是否反转（[rx,ry,rz] <-> [rz,ry,rx]）。
    - auto_swap_rpy_order: 抽样用 IK 可达性自动判断是否需要反转。

    轨迹规划（传入 spline_own.build_trajectory_plan）
    - cmdt_s: ServoCart 下发周期（cmdT），单位 s。
    - speed_mm_s: 期望沿路径最大速度（仍会被姿态/加速度约束压低），单位 mm/s。
    - chord_shrink_alpha: 弦线收缩比例（0 不收缩）。
    - max_ori_step_deg: 每个 cmdT 内允许的最大姿态变化量，单位 deg。
    - blend_mode: 转角几何模式："zone" 或 "fillet"。
    - fillet_radius_mm / zone_radius_mm: 转角圆角/交融区半径，单位 mm。
    - min_turn_deg: 只有偏离直线(180deg)足够大才认为是拐点。
    - max_centripetal_accel_mm_s2 / centripetal_accel_safety: 向心加速度限速。
    - max_tangential_accel_mm_s2: 切向加速度限速。
    - endpoint_speed_mm_s: 起终点速度上限。
    - corner_radius_safety / min_straight_len_mm: 几何交融的安全系数与最小直线段保留长度。
    - max_target_points / warn_if_targets_over: 下发点数上限与提示阈值。

    IK 预检（可选）
    - enable_ik_check: 是否对 targets 做抽样逆解预检。
    - ik_check_stride: 抽样步长（每隔多少点检查一次）。
    - ik_step_warn_deg: 单轴步进超过该阈值时给出警告。

    ServoCart 队列节流
    - queue_low_watermark / queue_high_watermark: MotionQueueLen 水位阈值。
    - queue_guard: 高水位预留裕量，防止超过 SDK 队列。
    - queue_prefill_target: 启动阶段预填充目标水位。
    - queue_poll_period_s: MotionQueueLen 轮询周期。
    - log_queue_len: 是否打印 MotionQueueLen。

    输出与行为
    - verbose: 是否打印关键过程日志。
    """

    robot_ip: str = "192.168.57.2"
    tool: int = 1
    user: int = 0
    points: list[list[float]] = field(default_factory=list)

    swap_rpy_order: bool = False
    auto_swap_rpy_order: bool = False

    cmdt_s: float = 0.004
    speed_mm_s: float = 450.0

    chord_shrink_alpha: float = 0.0
    max_ori_step_deg: float = 0.8

    blend_mode: str = "zone"
    fillet_radius_mm: float = 0.0
    zone_radius_mm: float = 0.0
    min_turn_deg: float = 10.0

    max_centripetal_accel_mm_s2: float = 2000.0
    centripetal_accel_safety: float = 0.9
    max_tangential_accel_mm_s2: float = 2000.0
    endpoint_speed_mm_s: float = 100.0
    corner_radius_safety: float = 1.0
    min_straight_len_mm: float = 0.0

    max_target_points: int = 8000
    warn_if_targets_over: int = 6000

    enable_ik_check: bool = False
    ik_check_stride: int = 25
    ik_step_warn_deg: float = 45.0

    queue_low_watermark: int = 5
    queue_high_watermark: int = 8
    queue_guard: int = 2
    queue_prefill_target: int = 6
    queue_poll_period_s: float = 0.008
    log_queue_len: bool = False

    verbose: bool = True


@dataclass(slots=True)
class ServoCartStats:
    """一次 ServoCart 运行的统计信息。"""

    plan: TrajectoryPlan
    target_count: int
    track_len_mm: float
    expected_time_s: float
    expected_avg_speed_mm_s: float
    servo_time_s: float
    actual_avg_speed_mm_s: float
    move_to_start_err: int
    servo_err: int
    completed: bool


@dataclass(slots=True)
class QueueSampler:
    robot: object
    poll_s: float
    last_t: float = 0.0
    last_err: int = -1
    last_len: int = 0

    def get(self) -> tuple[int, int]:
        now = time.perf_counter()
        if (now - self.last_t) >= float(self.poll_s):
            self.last_t = now
            self.last_err, self.last_len = get_queue_len(self.robot)
        return int(self.last_err), int(self.last_len)


def normalize_points(points: Sequence[Sequence[float]] | Iterable[Sequence[float]], *, swap_rpy_order: bool) -> list[list[float]]:
    """将输入点位整形为 list[list[float]]，并可选反转姿态三轴顺序。"""

    if points is None:
        raise RuntimeError("points 不能为空")

    poses: list[list[float]] = []
    for idx, point in enumerate(list(points)):
        if point is None:
            raise RuntimeError(f"points[{idx}] 为空")
        values = list(point)
        if len(values) < 6:
            raise RuntimeError(f"points[{idx}] 长度不足 6: {values!r}")
        pose = [float(v) for v in values[:6]]
        if bool(swap_rpy_order):
            pose[3:] = list(reversed(pose[3:]))
        poses.append(pose)

    if len(poses) < 2:
        raise RuntimeError("有效点位不足（至少 2 个）")

    return poses


def reverse_pose_rpy(poses: list[list[float]]) -> list[list[float]]:
    return [pose[:3] + list(reversed(pose[3:])) for pose in poses]


def count_reachable_samples(robot, poses: list[list[float]]) -> tuple[int, list[int]]:
    if not poses:
        return 0, []

    sample_indices = sorted({0, max(0, len(poses) // 2), len(poses) - 1})
    reachable = 0
    failed: list[int] = []
    for index in sample_indices:
        err, _ = robot.GetInverseKin(0, poses[index], -1)
        if int(err) == 0:
            reachable += 1
        else:
            failed.append(index)
    return reachable, failed


def resolve_pose_order(robot, poses: list[list[float]], *, swap_rpy_order: bool, auto_swap_rpy_order: bool, verbose: bool) -> list[list[float]]:
    if bool(swap_rpy_order) or not bool(auto_swap_rpy_order):
        return poses

    original_ok, original_failed = count_reachable_samples(robot, poses)
    reversed_poses = reverse_pose_rpy(poses)
    reversed_ok, reversed_failed = count_reachable_samples(robot, reversed_poses)

    if reversed_ok > original_ok:
        if verbose:
            print(
                "检测到点位文件姿态顺序更像 rz, ry, rx，已自动切换为反转姿态顺序；"
                f"原顺序可达样本={original_ok}, 反转后可达样本={reversed_ok}"
            )
        return reversed_poses

    if verbose:
        if original_ok == 0 and reversed_ok == 0:
            print("警告: 原顺序和反转顺序的抽样点都不可达，请检查点位文件本身")
        elif original_failed:
            print(f"提示: 原顺序存在不可达抽样点 {original_failed}，但仍保留原顺序")
        elif reversed_failed and reversed_ok == original_ok:
            print(f"提示: 反转姿态顺序也存在不可达抽样点 {reversed_failed}")

    return poses


def get_queue_len(robot) -> tuple[int, int]:
    try:
        err, length = robot.GetMotionQueueLength()
        return int(err), int(length)
    except Exception:
        return -1, 0


def wrap_deg(delta_deg: float) -> float:
    delta = float(delta_deg) % 360.0
    if delta > 180.0:
        delta -= 360.0
    return delta


def max_joint_step_deg(a: list[float], b: list[float]) -> float:
    return max(abs(wrap_deg(float(b[idx]) - float(a[idx]))) for idx in range(6))


def send_servo(robot, target: list[float], *, cmdt_s: float) -> int:
    return int(
        robot.ServoCart(
            mode=0,
            desc_pos=target,
            pos_gain=[1.0] * 6,
            acc=0.0,
            vel=0.0,
            cmdT=float(cmdt_s),
            filterT=0.0,
            gain=0.0,
        )
    )


def precheck_targets(robot, targets: list[list[float]], *, stride: int, ik_step_warn_deg: float, verbose: bool) -> None:
    stride = max(1, int(stride))
    joint_err, joint_ref = robot.GetActualJointPosDegree()
    if int(joint_err) != 0 or not joint_ref:
        if verbose:
            print("逆解预检跳过：未获取到当前关节位置")
        return

    indices = list(range(0, len(targets), stride))
    if not indices or indices[-1] != len(targets) - 1:
        indices.append(len(targets) - 1)

    checked = 0
    max_step = 0.0
    for index in indices:
        err, joint = robot.GetInverseKinRef(0, desc_pos=targets[index], joint_pos_ref=joint_ref)
        if int(err) != 0 or not joint:
            raise RuntimeError(f"逆解预检失败：target[{index}] error={err}")
        joint = [float(value) for value in joint]
        max_step = max(max_step, max_joint_step_deg(list(joint_ref), joint))
        joint_ref = joint
        checked += 1

    if verbose:
        print(f"逆解预检: 抽样点数={checked}, 最大单轴步进={max_step:.3f} deg")
        if max_step > float(ik_step_warn_deg):
            print(f"警告: 关节步进偏大，最大单轴步进超过 {float(ik_step_warn_deg):.3f} deg")


def follow_targets(
    robot,
    targets: list[list[float]],
    *,
    cmdt_s: float,
    queue_low_watermark: int,
    queue_high_watermark: int,
    queue_guard: int,
    queue_prefill_target: int,
    queue_poll_period_s: float,
    log_queue_len: bool,
) -> int:
    start_err = int(robot.ServoMoveStart())
    if start_err != 0:
        return start_err

    target_index = 0
    result = 0
    send_limit = max(0, int(queue_high_watermark) - int(queue_guard))
    sampler = QueueSampler(robot=robot, poll_s=max(float(queue_poll_period_s), float(cmdt_s)))

    try:
        prefill_mode = True
        next_send_t = time.perf_counter()
        while target_index < len(targets) and result == 0:
            queue_err, queue_len = sampler.get()
            queue_len = queue_len if queue_err == 0 else 0
            low_mark = int(queue_prefill_target) if prefill_mode else int(queue_low_watermark)
            if prefill_mode and queue_len >= int(queue_prefill_target):
                prefill_mode = False

            burst = 1 if queue_len >= low_mark else min(low_mark - queue_len + 1, 10)
            for _ in range(burst):
                if target_index >= len(targets) or result != 0:
                    break

                queue_err, queue_len = sampler.get()
                queue_len = queue_len if queue_err == 0 else int(queue_high_watermark)
                while queue_len > send_limit:
                    time.sleep(min(float(cmdt_s), float(sampler.poll_s)))
                    queue_err, queue_len = sampler.get()
                    queue_len = queue_len if queue_err == 0 else int(queue_high_watermark)

                if log_queue_len:
                    print(f"MotionQueueLen={queue_len}")

                result = send_servo(robot, targets[target_index], cmdt_s=cmdt_s)
                target_index += 1

            if burst > 1:
                next_send_t = time.perf_counter()
                continue

            next_send_t += float(cmdt_s)
            remain = next_send_t - time.perf_counter()
            if remain > 0.0:
                time.sleep(remain)
    finally:
        end_err = int(robot.ServoMoveEnd())
        if result == 0 and end_err != 0:
            result = end_err

    return int(result)


def run_servo_spline(
    robot,
    targets: list[list[float]],
    *,
    cmdt_s: float,
    queue_low_watermark: int,
    queue_high_watermark: int,
    queue_guard: int,
    queue_prefill_target: int,
    queue_poll_period_s: float,
    log_queue_len: bool,
) -> int:
    """纯样条伺服下发：不包含 MoveCart、回 HOME、读文件等任何业务动作。"""

    if not targets:
        raise RuntimeError("targets 为空")
    return int(
        follow_targets(
            robot,
            targets,
            cmdt_s=float(cmdt_s),
            queue_low_watermark=int(queue_low_watermark),
            queue_high_watermark=int(queue_high_watermark),
            queue_guard=int(queue_guard),
            queue_prefill_target=int(queue_prefill_target),
            queue_poll_period_s=float(queue_poll_period_s),
            log_queue_len=bool(log_queue_len),
        )
    )


def connect_robot(robot_ip: str):
    """连接 Fairino 控制器并返回 robot 对象。"""

    if Robot is None:
        raise RuntimeError("未能导入 Fairino SDK（fairino390.linux.fairino.Robot）。请确认 SDK 已安装且路径正确。")
    return Robot.RPC(str(robot_ip))


def plan_from_points(robot, config: ServoCartConfig) -> tuple[TrajectoryPlan, list[list[float]]]:
    """从点位列表生成轨迹计划与 ServoCart targets。"""

    raw_points = normalize_points(config.points, swap_rpy_order=bool(config.swap_rpy_order))
    raw_points = resolve_pose_order(
        robot,
        raw_points,
        swap_rpy_order=config.swap_rpy_order,
        auto_swap_rpy_order=config.auto_swap_rpy_order,
        verbose=config.verbose,
    )

    min_z = min(point[2] for point in raw_points)
    path_points = raw_points
    if float(config.chord_shrink_alpha) > 0.0 and len(raw_points) >= 3:
        path_points = shrink_points_to_chord(raw_points, alpha=float(config.chord_shrink_alpha), clamp_to_segment=True)
        if config.verbose:
            print(f"弦线收缩 alpha={float(config.chord_shrink_alpha):.3f}")

    plan = build_trajectory_plan(
        path_points,
        cmdt_s=float(config.cmdt_s),
        fillet_radius_mm=float(config.fillet_radius_mm),
        zone_radius_mm=float(config.zone_radius_mm),
        blend_mode=str(config.blend_mode),
        min_turn_deg=float(config.min_turn_deg),
        corner_radius_safety=float(config.corner_radius_safety),
        min_straight_len_mm=float(config.min_straight_len_mm),
        speed_mm_s=float(config.speed_mm_s),
        min_z_mm=float(min_z),
        max_target_points=int(config.max_target_points),
        max_ori_step_deg=float(config.max_ori_step_deg),
        max_centripetal_accel_mm_s2=float(config.max_centripetal_accel_mm_s2),
        centripetal_accel_safety=float(config.centripetal_accel_safety),
        max_tangential_accel_mm_s2=float(config.max_tangential_accel_mm_s2),
        endpoint_speed_mm_s=float(config.endpoint_speed_mm_s),
    )
    targets = plan.targets
    if not targets:
        raise RuntimeError("轨迹生成失败：未生成任何目标点")

    return plan, targets


def run_servo_cart(config: ServoCartConfig, *, robot=None) -> ServoCartStats:
    """执行一次 ServoCart 轨迹下发。

    Args:
        config: 配置（默认值来自原 `servo.py`）。
        robot: 可选，外部已连接的 robot 对象；若不传则内部按 config.robot_ip 连接并在结束时 CloseRPC。

    Returns:
        ServoCartStats: 统计信息（包含轨迹 plan、速度、耗时、错误码）。
    """

    start_t = time.perf_counter()
    created_robot = False
    if robot is None:
        robot = connect_robot(config.robot_ip)
        created_robot = True

    try:
        plan, targets = plan_from_points(robot, config)

        track_len = float(plan.target_length_mm)
        expected_time = len(targets) * float(plan.used_cmdt_s)
        expected_speed = track_len / expected_time if expected_time > 1e-9 else 0.0

        if config.verbose:
            try:
                print(f"SDK版本: {robot.GetSDKVersion()}")
            except Exception:
                pass
            print(f"几何模式: {plan.geometry_mode}")
            print(f"参考轨迹长度: {float(plan.reference_length_mm):.3f} mm")
            print(f"目标点数: {len(targets)}")
            print(f"轨迹长度: {track_len:.3f} mm")
            print(f"请求 cmdT: {float(plan.requested_cmdt_s):.6f} s")
            print(f"实际 cmdT: {float(plan.used_cmdt_s):.6f} s")
            print(f"点数上限: {int(plan.max_target_points)}")
            print(f"姿态步长上限: {float(plan.max_ori_step_deg):.3f} deg")
            print(f"目标速度: {float(config.speed_mm_s):.3f} mm/s")
            print(f"预计时间: {expected_time:.3f} s")
            print(f"预计平均速度: {expected_speed:.3f} mm/s")
            if len(targets) >= int(config.warn_if_targets_over):
                print(f"提示: 当前目标点数较多，已接近上限 {int(config.max_target_points)}")

            print("提示: SDK 不再 MoveCart 到起点；请在调用前自行 MoveCart 到 plan.geometry_points[0]")

        if bool(config.enable_ik_check):
            precheck_targets(
                robot,
                targets,
                stride=int(config.ik_check_stride),
                ik_step_warn_deg=float(config.ik_step_warn_deg),
                verbose=bool(config.verbose),
            )

        servo_start_t = time.perf_counter()
        servo_err = run_servo_spline(
            robot,
            targets,
            cmdt_s=float(plan.used_cmdt_s),
            queue_low_watermark=int(config.queue_low_watermark),
            queue_high_watermark=int(config.queue_high_watermark),
            queue_guard=int(config.queue_guard),
            queue_prefill_target=int(config.queue_prefill_target),
            queue_poll_period_s=float(config.queue_poll_period_s),
            log_queue_len=bool(config.log_queue_len),
        )
        servo_time = time.perf_counter() - servo_start_t
        total_time = time.perf_counter() - start_t
        actual_speed = track_len / servo_time if servo_time > 1e-9 else 0.0
        completed = (int(servo_err) == 0)

        if config.verbose:
            print(f"ServoCart 结束，错误码: {servo_err}")
            print(f"伺服时间: {servo_time:.3f} s")
            print(f"实际平均速度: {actual_speed:.3f} mm/s")
            print(f"运行总时间: {total_time:.3f} s")
            if completed:
                print("Servo 正常结束")

        return ServoCartStats(
            plan=plan,
            target_count=len(targets),
            track_len_mm=track_len,
            expected_time_s=expected_time,
            expected_avg_speed_mm_s=expected_speed,
            servo_time_s=float(servo_time),
            actual_avg_speed_mm_s=float(actual_speed),
            move_to_start_err=0,
            servo_err=int(servo_err),
            completed=bool(completed),
        )
    finally:
        if created_robot:
            try:
                robot.CloseRPC()
            except Exception:
                pass


def run_servo_cart_simple(
    *,
    points: Sequence[Sequence[float]] | None = None,
    robot_ip: str | None = None,
    tool: int | None = None,
    user: int | None = None,
    cmdt_s: float | None = None,
    speed_mm_s: float | None = None,
    max_ori_step_deg: float | None = None,
    swap_rpy_order: bool | None = None,
    auto_swap_rpy_order: bool | None = None,
    verbose: bool | None = None,
    robot=None,
) -> ServoCartStats:
    """ServoCart SDK（简化调用接口）。

    原型（SDK 对外）：

        run_servo_cart_simple(
            points=None,
            robot_ip=None,
            tool=None,
            user=None,
            cmdt_s=None,
            speed_mm_s=None,
            max_ori_step_deg=None,
            swap_rpy_order=None,
            auto_swap_rpy_order=None,
            verbose=None,
            robot=None,
        ) -> ServoCartStats

    描述：
    - 面向“只想调 3 个关键参数”的调用方式：`cmdt_s` / `speed_mm_s` / `max_ori_step_deg`。
    - 其它所有参数都沿用 `ServoCartConfig()` 的默认值（也就是你原始 `servo.py` 的默认参数）。
    - 若传入 `robot`，则复用外部连接（SDK 不会 CloseRPC）。
      若不传入 `robot`，SDK 会按 `robot_ip` 连接并在结束时 CloseRPC。

    必选参数：
    - points: 点位列表（每个点至少 6 个元素 [x,y,z,rx,ry,rz]）。

    可选参数（你关心的 3 个）：
    - cmdt_s: 指令下发周期（ServoCart 的 cmdT），单位 s；不传则用默认 0.004。
    - speed_mm_s: 沿路径最大速度上限，单位 mm/s；不传则用默认 450。
    - max_ori_step_deg: 每个 cmdT 内允许的最大姿态变化量，单位 deg；不传则用默认 0.8。

    其余可选参数（常用但不是必须）：
    - robot_ip/tool/user: 覆盖连接与坐标系。
    - swap_rpy_order/auto_swap_rpy_order: 姿态三轴顺序处理。
    - verbose: 控制打印。

    返回值：
    - ServoCartStats：包含 `servo_err`（0 成功，否则失败错误码）、耗时、平均速度、轨迹 plan 等信息。

    参考（底层调用的 Fairino SDK 原型，供对照）：

        ServoCart(
            mode,
            desc_pos,
            pos_gain,
            acc=0.0,
            vel=0.0,
            cmdT=0.008,
            filterT=0.0,
            gain=0.0,
        ) -> errcode

    注：本项目当前仅显式使用 `cmdT`（即本函数的 `cmdt_s`），其它参数保持 0.0。
    """

    config = ServoCartConfig()
    if points is not None:
        config.points = [list(map(float, p[:6])) for p in points]
    if robot_ip is not None:
        config.robot_ip = str(robot_ip)
    if tool is not None:
        config.tool = int(tool)
    if user is not None:
        config.user = int(user)

    if cmdt_s is not None:
        config.cmdt_s = float(cmdt_s)
    if speed_mm_s is not None:
        config.speed_mm_s = float(speed_mm_s)
    if max_ori_step_deg is not None:
        config.max_ori_step_deg = float(max_ori_step_deg)

    if swap_rpy_order is not None:
        config.swap_rpy_order = bool(swap_rpy_order)
    if auto_swap_rpy_order is not None:
        config.auto_swap_rpy_order = bool(auto_swap_rpy_order)
    if verbose is not None:
        config.verbose = bool(verbose)

    return run_servo_cart(config, robot=robot)


def ServoCartSDK(
    points: Sequence[Sequence[float]],
    *,
    robot_ip: str = ServoCartConfig.robot_ip,
    tool: int = ServoCartConfig.tool,
    user: int = ServoCartConfig.user,
    swap_rpy_order: bool = ServoCartConfig.swap_rpy_order,
    auto_swap_rpy_order: bool = ServoCartConfig.auto_swap_rpy_order,
    cmdt_s: float = ServoCartConfig.cmdt_s,
    speed_mm_s: float = ServoCartConfig.speed_mm_s,
    chord_shrink_alpha: float = ServoCartConfig.chord_shrink_alpha,
    max_ori_step_deg: float = ServoCartConfig.max_ori_step_deg,
    blend_mode: str = ServoCartConfig.blend_mode,
    fillet_radius_mm: float = ServoCartConfig.fillet_radius_mm,
    zone_radius_mm: float = ServoCartConfig.zone_radius_mm,
    min_turn_deg: float = ServoCartConfig.min_turn_deg,
    max_centripetal_accel_mm_s2: float = ServoCartConfig.max_centripetal_accel_mm_s2,
    centripetal_accel_safety: float = ServoCartConfig.centripetal_accel_safety,
    max_tangential_accel_mm_s2: float = ServoCartConfig.max_tangential_accel_mm_s2,
    endpoint_speed_mm_s: float = ServoCartConfig.endpoint_speed_mm_s,
    corner_radius_safety: float = ServoCartConfig.corner_radius_safety,
    min_straight_len_mm: float = ServoCartConfig.min_straight_len_mm,
    max_target_points: int = ServoCartConfig.max_target_points,
    warn_if_targets_over: int = ServoCartConfig.warn_if_targets_over,
    enable_ik_check: bool = ServoCartConfig.enable_ik_check,
    ik_check_stride: int = ServoCartConfig.ik_check_stride,
    ik_step_warn_deg: float = ServoCartConfig.ik_step_warn_deg,
    queue_low_watermark: int = ServoCartConfig.queue_low_watermark,
    queue_high_watermark: int = ServoCartConfig.queue_high_watermark,
    queue_guard: int = ServoCartConfig.queue_guard,
    queue_prefill_target: int = ServoCartConfig.queue_prefill_target,
    queue_poll_period_s: float = ServoCartConfig.queue_poll_period_s,
    log_queue_len: bool = ServoCartConfig.log_queue_len,
    verbose: bool = ServoCartConfig.verbose,
    robot=None,
) -> int:
    """ServoCart SDK（ServoJ 风格：原型 + 必选参数 + 默认参数 + 返回值）。

    原型：

        ServoCartSDK(
            points,
            robot_ip="192.168.57.2",
            tool=1,
            user=0,
            swap_rpy_order=False,
            auto_swap_rpy_order=False,
            cmdt_s=0.004,
            speed_mm_s=450,
            chord_shrink_alpha=0.0,
            max_ori_step_deg=0.8,
            blend_mode="zone",
            fillet_radius_mm=0.0,
            zone_radius_mm=0.0,
            min_turn_deg=10.0,
            max_centripetal_accel_mm_s2=2000,
            centripetal_accel_safety=0.9,
            max_tangential_accel_mm_s2=2000,
            endpoint_speed_mm_s=100,
            corner_radius_safety=1,
            min_straight_len_mm=0.0,
            max_target_points=8000,
            warn_if_targets_over=6000,
            enable_ik_check=False,
            ik_check_stride=25,
            ik_step_warn_deg=45.0,
            queue_low_watermark=5,
            queue_high_watermark=8,
            queue_guard=2,
            queue_prefill_target=6,
            queue_poll_period_s=0.008,
            log_queue_len=False,
            verbose=True,
            robot=None,
        ) -> errcode

    描述：
    - ServoCart 的“读点 -> 轨迹规划 -> 下发”一体化封装。
    - 你在原 `servo.py` 能改的参数，这里全部以函数参数形式暴露（都有默认值）。
    - 默认值来源：`ServoCartConfig` 的默认字段值（等价于你原先脚本常量区）。
    - 若传入 `robot`，复用外部连接（SDK 不会 CloseRPC）；否则内部按 `robot_ip` 连接并在结束时 CloseRPC。

    必选参数：
        - points: 点位列表（每个点至少 6 个元素 [x,y,z,rx,ry,rz]）。

    默认参数（节选/常用）：
    - cmdt_s: 指令下发周期（Fairino ServoCart 的 cmdT），单位 s。
    - speed_mm_s: 末端沿路径最大速度上限，单位 mm/s。
    - max_ori_step_deg: 每个 cmdT 内允许的最大姿态变化量，单位 deg。
    - queue_*: ServoCart MotionQueueLen 节流相关阈值。

    返回值：
    - errcode：成功返回 0；失败返回对应错误码（下发前 MoveCart/ServoCart/ServoMoveStart 等任何一步的错误码）。
      如果你需要更多统计信息（预计/实际速度、耗时、轨迹 plan），请用 `run_servo_cart(...)`，它返回 `ServoCartStats`。
    """

    config = ServoCartConfig(
        robot_ip=str(robot_ip),
        tool=int(tool),
        user=int(user),
        points=[list(map(float, p[:6])) for p in points],
        swap_rpy_order=bool(swap_rpy_order),
        auto_swap_rpy_order=bool(auto_swap_rpy_order),
        cmdt_s=float(cmdt_s),
        speed_mm_s=float(speed_mm_s),
        chord_shrink_alpha=0.0,
        max_ori_step_deg=float(max_ori_step_deg),
        blend_mode=str(blend_mode),
        fillet_radius_mm=float(fillet_radius_mm),
        zone_radius_mm=float(zone_radius_mm),
        min_turn_deg=float(min_turn_deg),
        max_centripetal_accel_mm_s2=float(max_centripetal_accel_mm_s2),
        centripetal_accel_safety=float(centripetal_accel_safety),
        max_tangential_accel_mm_s2=float(max_tangential_accel_mm_s2),
        endpoint_speed_mm_s=float(endpoint_speed_mm_s),
        corner_radius_safety=float(corner_radius_safety),
        min_straight_len_mm=float(min_straight_len_mm),
        max_target_points=int(max_target_points),
        warn_if_targets_over=int(warn_if_targets_over),
        enable_ik_check=bool(enable_ik_check),
        ik_check_stride=int(ik_check_stride),
        ik_step_warn_deg=float(ik_step_warn_deg),
        queue_low_watermark=int(queue_low_watermark),
        queue_high_watermark=int(queue_high_watermark),
        queue_guard=int(queue_guard),
        queue_prefill_target=int(queue_prefill_target),
        queue_poll_period_s=float(queue_poll_period_s),
        log_queue_len=bool(log_queue_len),
        verbose=bool(verbose),
    )
    stats = run_servo_cart(config, robot=robot)
    return int(stats.servo_err)


__all__ = [
    "ServoCartConfig",
    "ServoCartStats",
    "ServoCartSDK",
    "connect_robot",
    "normalize_points",
    "plan_from_points",
    "run_servo_spline",
    "run_servo_cart",
    "run_servo_cart_simple",
]
