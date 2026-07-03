"""统一轨迹规划器。"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Literal, Sequence

from calculate import compute_fillet_profile, compute_zone_profile, dist_xyz


CornerMode = Literal["auto", "zone", "fillet", "linear"]


def _rotation_angle_between_mats_deg(a: list[list[float]], b: list[list[float]]) -> float:
    rel = _mat_mul(_mat_transpose(a), b)
    trace = rel[0][0] + rel[1][1] + rel[2][2]
    cos_angle = _clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _triangle_curvature_mm_inv(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = _sub(b, a)
    ac = _sub(c, a)
    area2 = _norm(_cross(ab, ac))
    if area2 < 1e-12:
        return 0.0
    lab = _dist_xyz(a, b)
    lbc = _dist_xyz(b, c)
    lac = _dist_xyz(a, c)
    denom = lab * lbc * lac
    if denom < 1e-12:
        return 0.0
    # curvature kappa = 1/R = 4A/(abc); with area2=2A => kappa=2*area2/(abc)
    return (2.0 * area2) / denom


def _enforce_accel_limits(
    s: Sequence[float],
    v_max: Sequence[float],
    *,
    a_t_max: float,
    v_start: float,
    v_end: float,
) -> list[float]:
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [min(float(v_max[0]), float(v_start), float(v_end))]

    v = [max(0.0, float(value)) for value in v_max]
    v[0] = min(v[0], max(0.0, float(v_start)))
    v[-1] = min(v[-1], max(0.0, float(v_end)))

    a = max(1e-6, float(a_t_max))

    # forward pass (acceleration)
    for i in range(n - 1):
        ds = float(s[i + 1]) - float(s[i])
        if ds <= 1e-12:
            v[i + 1] = min(v[i + 1], v[i])
            continue
        v_allowed = math.sqrt(max(0.0, v[i] * v[i] + 2.0 * a * ds))
        v[i + 1] = min(v[i + 1], v_allowed)

    # backward pass (deceleration)
    for i in range(n - 2, -1, -1):
        ds = float(s[i + 1]) - float(s[i])
        if ds <= 1e-12:
            v[i] = min(v[i], v[i + 1])
            continue
        v_allowed = math.sqrt(max(0.0, v[i + 1] * v[i + 1] + 2.0 * a * ds))
        v[i] = min(v[i], v_allowed)

    return v


def _build_time_table(s: Sequence[float], v: Sequence[float]) -> list[float]:
    n = len(s)
    if n == 0:
        return []
    t = [0.0] * n
    for i in range(n - 1):
        ds = float(s[i + 1]) - float(s[i])
        if ds <= 0.0:
            t[i + 1] = t[i]
            continue
        v_avg = 0.5 * (max(0.0, float(v[i])) + max(0.0, float(v[i + 1])))
        v_avg = max(1e-3, v_avg)
        t[i + 1] = t[i] + ds / v_avg
    return t


def _invert_monotone_table(x: Sequence[float], y: Sequence[float], yq: float) -> float:
    # Given monotone increasing y(x), query x(yq) via linear interpolation.
    if not x:
        return 0.0
    if yq <= float(y[0]):
        return float(x[0])
    if yq >= float(y[-1]):
        return float(x[-1])
    idx = bisect.bisect_right(y, float(yq)) - 1
    idx = max(0, min(idx, len(y) - 2))
    y0 = float(y[idx])
    y1 = float(y[idx + 1])
    if y1 <= y0 + 1e-12:
        return float(x[idx])
    r = (float(yq) - y0) / (y1 - y0)
    return float(x[idx]) + (float(x[idx + 1]) - float(x[idx])) * r


@dataclass
class TrajectoryPlan:
    geometry_mode: str
    geometry_points: list[list[float]]
    targets: list[list[float]]
    reference_length_mm: float
    target_length_mm: float
    requested_cmdt_s: float
    used_cmdt_s: float
    max_target_points: int
    max_ori_step_deg: float


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(_dot(vec, vec))


def _dist_xyz(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mul(a: Sequence[float], scale: float) -> list[float]:
    return [a[0] * scale, a[1] * scale, a[2] * scale]


def _unit(vec: Sequence[float], eps: float = 1e-9) -> list[float]:
    length = _norm(vec)
    if length < eps:
        return [0.0, 0.0, 0.0]
    return [vec[0] / length, vec[1] / length, vec[2] / length]


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _mat_transpose(mat: list[list[float]]) -> list[list[float]]:
    return [
        [mat[0][0], mat[1][0], mat[2][0]],
        [mat[0][1], mat[1][1], mat[2][1]],
        [mat[0][2], mat[1][2], mat[2][2]],
    ]


def _rodrigues(axis_unit: Sequence[float], angle_rad: float) -> list[list[float]]:
    x, y, z = axis_unit
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    one_minus = 1.0 - cos_a
    return [
        [
            cos_a + x * x * one_minus,
            x * y * one_minus - z * sin_a,
            x * z * one_minus + y * sin_a,
        ],
        [
            y * x * one_minus + z * sin_a,
            cos_a + y * y * one_minus,
            y * z * one_minus - x * sin_a,
        ],
        [
            z * x * one_minus - y * sin_a,
            z * y * one_minus + x * sin_a,
            cos_a + z * z * one_minus,
        ],
    ]


def _rotate_vector(vec: Sequence[float], axis_unit: Sequence[float], angle_rad: float) -> list[float]:
    rot = _rodrigues(axis_unit, angle_rad)
    return [
        rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
        rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
        rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2],
    ]


def _euler_to_matrix_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> list[list[float]]:
    rx = math.radians(float(rx_deg))
    ry_rad = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))

    cx = math.cos(rx)
    sx = math.sin(rx)
    cy = math.cos(ry_rad)
    sy = math.sin(ry_rad)
    cz = math.cos(rz)
    sz = math.sin(rz)

    rx_mat = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    ry_mat = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    rz_mat = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return _mat_mul(_mat_mul(rz_mat, ry_mat), rx_mat)


def _matrix_to_euler_deg(rot: list[list[float]]) -> list[float]:
    r20 = _clamp(rot[2][0], -1.0, 1.0)
    ry_rad = math.asin(-r20)
    cos_ry = math.cos(ry_rad)

    if abs(cos_ry) > 1e-8:
        rx = math.atan2(rot[2][1], rot[2][2])
        rz = math.atan2(rot[1][0], rot[0][0])
    else:
        rx = 0.0
        if r20 <= -0.999999:
            ry_rad = math.pi / 2.0
            rz = math.atan2(-rot[0][1], rot[1][1])
        else:
            ry_rad = -math.pi / 2.0
            rz = math.atan2(rot[0][1], rot[1][1])

    return [math.degrees(rx), math.degrees(ry_rad), math.degrees(rz)]


def _rotation_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    mat_a = _euler_to_matrix_deg(a[3], a[4], a[5])
    mat_b = _euler_to_matrix_deg(b[3], b[4], b[5])
    rel = _mat_mul(_mat_transpose(mat_a), mat_b)
    trace = rel[0][0] + rel[1][1] + rel[2][2]
    cos_angle = _clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _interpolate_rotation_matrix(start_rot: list[list[float]], end_rot: list[list[float]], t: float) -> list[list[float]]:
    alpha = _clamp(float(t), 0.0, 1.0)
    rel = _mat_mul(_mat_transpose(start_rot), end_rot)
    trace = rel[0][0] + rel[1][1] + rel[2][2]
    cos_angle = _clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return [row[:] for row in start_rot]

    axis = [
        rel[2][1] - rel[1][2],
        rel[0][2] - rel[2][0],
        rel[1][0] - rel[0][1],
    ]
    axis_norm = _norm(axis)
    if axis_norm < 1e-9:
        return [row[:] for row in start_rot]
    axis_unit = [axis[0] / axis_norm, axis[1] / axis_norm, axis[2] / axis_norm]
    return _mat_mul(start_rot, _rodrigues(axis_unit, angle * alpha))


def _pose_with_xyz_and_rot(xyz: Sequence[float], rot: list[list[float]]) -> list[float]:
    rpy = _matrix_to_euler_deg(rot)
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), rpy[0], rpy[1], rpy[2]]


def _interpolate_pose(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    alpha = _clamp(float(t), 0.0, 1.0)
    xyz = [float(a[i]) + (float(b[i]) - float(a[i])) * alpha for i in range(3)]
    rot_a = _euler_to_matrix_deg(a[3], a[4], a[5])
    rot_b = _euler_to_matrix_deg(b[3], b[4], b[5])
    rot = _interpolate_rotation_matrix(rot_a, rot_b, alpha)
    return _pose_with_xyz_and_rot(xyz, rot)


def _append_pose_unique(out: list[list[float]], pose: Sequence[float]) -> None:
    new_pose = [float(value) for value in pose[:6]]
    if not out:
        out.append(new_pose)
        return
    last = out[-1]
    if _dist_xyz(last, new_pose) < 1e-6 and _rotation_angle_deg(last, new_pose) < 1e-4:
        out[-1] = new_pose
        return
    out.append(new_pose)


def _normalize_points(points: Sequence[Sequence[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for raw_point in points:
        pose = [float(value) for value in raw_point[:6]]
        if not normalized:
            normalized.append(pose)
            continue
        xyz_step = _dist_xyz(normalized[-1], pose)
        rot_step = _rotation_angle_deg(normalized[-1], pose)
        if xyz_step < 1e-9 and rot_step < 1e-6:
            continue
        normalized.append(pose)
    if len(normalized) < 2:
        raise ValueError("有效点位不足，至少需要 2 个不同点")
    return normalized


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(dist_xyz(points[idx], points[idx + 1]) for idx in range(len(points) - 1))


def _build_axis(points: Sequence[Sequence[float]]) -> list[float]:
    axis = [0.0]
    for idx in range(1, len(points)):
        step = _dist_xyz(points[idx - 1], points[idx])
        if step < 1e-9:
            step = 1e-3
        axis.append(axis[-1] + step)
    return axis


def _estimate_speed_mm_s(
    points: Sequence[Sequence[float]],
    *,
    cmdt_s: float,
    seg_time_s: float,
    speed_mm_s: float | None,
) -> float:
    if speed_mm_s is not None and float(speed_mm_s) > 0.0:
        return float(speed_mm_s)
    total_length = _polyline_length(points)
    seg_time = float(seg_time_s) if float(seg_time_s) > 0.0 else 2.0
    total_time = max(float(cmdt_s), (len(points) - 1) * seg_time)
    if total_length <= 1e-9:
        return 1.0
    return total_length / total_time


class _QuinticSegment1D:
    def __init__(self, x0: float, v0: float, a0: float, x1: float, v1: float, a1: float, span: float):
        if span <= 0.0:
            raise ValueError("span must be > 0")
        self.x0 = float(x0)
        self.v0 = float(v0)
        self.a0 = float(a0)
        self.x1 = float(x1)
        self.v1 = float(v1)
        self.a1 = float(a1)
        self.span = float(span)

    def eval(self, local_s: float) -> float:
        tau = _clamp(float(local_s) / self.span, 0.0, 1.0)
        t2 = tau * tau
        t3 = t2 * tau
        t4 = t3 * tau
        t5 = t4 * tau
        h = self.span
        h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
        h10 = tau - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
        h20 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
        h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
        h11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
        h21 = 0.5 * t3 - t4 + 0.5 * t5
        return (
            h00 * self.x0
            + h10 * h * self.v0
            + h20 * h * h * self.a0
            + h01 * self.x1
            + h11 * h * self.v1
            + h21 * h * h * self.a1
        )


class _QuinticPath1D:
    def __init__(self, axis: Sequence[float], values: Sequence[float]):
        if len(axis) != len(values):
            raise ValueError("axis and values must have the same length")
        if len(axis) < 2:
            raise ValueError("at least two samples are required")
        self.axis = [float(value) for value in axis]
        self.values = [float(value) for value in values]
        self.v = self._estimate_first_derivatives()
        self.a = self._estimate_second_derivatives()
        self.segments = [
            _QuinticSegment1D(
                self.values[idx],
                self.v[idx],
                self.a[idx],
                self.values[idx + 1],
                self.v[idx + 1],
                self.a[idx + 1],
                self.axis[idx + 1] - self.axis[idx],
            )
            for idx in range(len(self.axis) - 1)
        ]

    def _estimate_first_derivatives(self) -> list[float]:
        n = len(self.axis)
        deriv = [0.0] * n
        if n == 2:
            span = self.axis[1] - self.axis[0]
            slope = 0.0 if span <= 0.0 else (self.values[1] - self.values[0]) / span
            return [slope, slope]
        for idx in range(n):
            if idx == 0:
                span = self.axis[1] - self.axis[0]
                deriv[idx] = 0.0 if span <= 0.0 else (self.values[1] - self.values[0]) / span
            elif idx == n - 1:
                span = self.axis[-1] - self.axis[-2]
                deriv[idx] = 0.0 if span <= 0.0 else (self.values[-1] - self.values[-2]) / span
            else:
                span = self.axis[idx + 1] - self.axis[idx - 1]
                deriv[idx] = 0.0 if span <= 0.0 else (self.values[idx + 1] - self.values[idx - 1]) / span
        return deriv

    def _estimate_second_derivatives(self) -> list[float]:
        n = len(self.axis)
        second = [0.0] * n
        for idx in range(1, n - 1):
            h0 = self.axis[idx] - self.axis[idx - 1]
            h1 = self.axis[idx + 1] - self.axis[idx]
            if h0 <= 0.0 or h1 <= 0.0:
                continue
            slope0 = (self.values[idx] - self.values[idx - 1]) / h0
            slope1 = (self.values[idx + 1] - self.values[idx]) / h1
            second[idx] = 2.0 * (slope1 - slope0) / (self.axis[idx + 1] - self.axis[idx - 1])
        return second

    def eval(self, axis_value: float) -> float:
        x = float(axis_value)
        if x <= self.axis[0]:
            idx = 0
        elif x >= self.axis[-1]:
            idx = len(self.axis) - 2
        else:
            idx = bisect.bisect_right(self.axis, x) - 1
        return self.segments[idx].eval(x - self.axis[idx])


def _build_position_paths(points: Sequence[Sequence[float]], axis: Sequence[float]) -> tuple[_QuinticPath1D, _QuinticPath1D, _QuinticPath1D]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    z_values = [point[2] for point in points]
    return _QuinticPath1D(axis, x_values), _QuinticPath1D(axis, y_values), _QuinticPath1D(axis, z_values)


def _evaluate_position(
    axis_value: float,
    *,
    x_path: _QuinticPath1D,
    y_path: _QuinticPath1D,
    z_path: _QuinticPath1D,
    min_z: float | None,
) -> list[float]:
    x = x_path.eval(axis_value)
    y = y_path.eval(axis_value)
    z = z_path.eval(axis_value)
    if min_z is not None:
        z = max(float(min_z), z)
    return [x, y, z]


def _evaluate_rotation(axis_value: float, axis: Sequence[float], rotations: Sequence[list[list[float]]]) -> list[list[float]]:
    if axis_value <= axis[0]:
        return [row[:] for row in rotations[0]]
    if axis_value >= axis[-1]:
        return [row[:] for row in rotations[-1]]
    idx = bisect.bisect_right(axis, axis_value) - 1
    idx = max(0, min(idx, len(axis) - 2))
    span = axis[idx + 1] - axis[idx]
    local_t = 0.0 if span <= 1e-12 else (axis_value - axis[idx]) / span
    return _interpolate_rotation_matrix(rotations[idx], rotations[idx + 1], local_t)


def _dense_axis_values(total_axis: float, target_step_mm: float, point_count: int) -> list[float]:
    if total_axis <= 1e-9:
        return [0.0]
    dense_step = min(2.0, max(0.2, target_step_mm / 4.0))
    dense_count = max(point_count * 8, int(math.ceil(total_axis / dense_step)) + 1)
    if dense_count < 2:
        dense_count = 2
    step = total_axis / float(dense_count - 1)
    return [step * idx for idx in range(dense_count)]


def _build_arc_table(points_xyz: Sequence[Sequence[float]]) -> list[float]:
    arc = [0.0]
    for idx in range(1, len(points_xyz)):
        arc.append(arc[-1] + _dist_xyz(points_xyz[idx - 1], points_xyz[idx]))
    return arc


def _find_axis_from_arc(target_arc: float, dense_arc: Sequence[float], dense_axis: Sequence[float]) -> float:
    if target_arc <= 0.0:
        return dense_axis[0]
    if target_arc >= dense_arc[-1]:
        return dense_axis[-1]
    idx = bisect.bisect_right(dense_arc, target_arc) - 1
    idx = max(0, min(idx, len(dense_arc) - 2))
    span = dense_arc[idx + 1] - dense_arc[idx]
    if span <= 1e-12:
        return dense_axis[idx]
    ratio = (target_arc - dense_arc[idx]) / span
    return dense_axis[idx] + (dense_axis[idx + 1] - dense_axis[idx]) * ratio


def _build_adaptive_targets(
    *,
    dense_axis: Sequence[float],
    dense_xyz: Sequence[Sequence[float]],
    dense_rotations: Sequence[list[list[float]]],
    target_step_mm: float,
    max_rot_step_deg: float,
) -> list[list[float]]:
    if not dense_axis:
        return []

    targets: list[list[float]] = []
    last_xyz = list(map(float, dense_xyz[0][:3]))
    last_rot = [row[:] for row in dense_rotations[0]]

    pos_limit = max(0.1, float(target_step_mm))
    rot_limit = max(0.05, float(max_rot_step_deg))

    for idx in range(1, len(dense_axis)):
        xyz = dense_xyz[idx]
        rot = dense_rotations[idx]
        pos_delta = _dist_xyz(last_xyz, xyz)
        rot_delta = _rotation_angle_deg(
            _pose_with_xyz_and_rot(last_xyz, last_rot),
            _pose_with_xyz_and_rot(xyz, rot),
        )
        if pos_delta >= pos_limit or rot_delta >= rot_limit:
            targets.append(_pose_with_xyz_and_rot(xyz, rot))
            last_xyz = list(map(float, xyz[:3]))
            last_rot = [row[:] for row in rot]

    final_pose = _pose_with_xyz_and_rot(dense_xyz[-1], dense_rotations[-1])
    if not targets:
        targets.append(final_pose)
    else:
        last_pose = targets[-1]
        if _dist_xyz(last_pose, final_pose) > 1e-6 or _rotation_angle_deg(last_pose, final_pose) > 1e-4:
            targets.append(final_pose)
    return targets


def shrink_points_to_chord(
    points: Sequence[Sequence[float]],
    *,
    alpha: float,
    clamp_to_segment: bool = True,
) -> list[list[float]]:
    pts = [list(map(float, point[:6])) for point in points]
    if len(pts) < 3:
        return pts
    strength = float(alpha)
    if strength <= 0.0:
        return pts
    start = pts[0][:3]
    end = pts[-1][:3]
    line = [end[i] - start[i] for i in range(3)]
    line_len = _norm(line)
    if line_len < 1e-9:
        return pts
    line_unit = [line[i] / line_len for i in range(3)]
    out = [pts[0][:]]
    for idx in range(1, len(pts) - 1):
        point = pts[idx]
        sp = [point[j] - start[j] for j in range(3)]
        proj_len = _dot(sp, line_unit)
        if clamp_to_segment:
            proj_len = _clamp(proj_len, 0.0, line_len)
        proj = [start[j] + line_unit[j] * proj_len for j in range(3)]
        new_xyz = [point[j] + (proj[j] - point[j]) * strength for j in range(3)]
        out.append([new_xyz[0], new_xyz[1], new_xyz[2], point[3], point[4], point[5]])
    out.append(pts[-1][:])
    return out


def _sample_line_segment(out: list[list[float]], start_pose: Sequence[float], end_pose: Sequence[float], max_step_mm: float) -> None:
    distance = _dist_xyz(start_pose, end_pose)
    steps = max(1, int(math.ceil(distance / max(0.5, float(max_step_mm)))))
    for idx in range(1, steps + 1):
        _append_pose_unique(out, _interpolate_pose(start_pose, end_pose, idx / float(steps)))


def _sample_arc_segment(
    out: list[list[float]],
    *,
    center: Sequence[float],
    start_xyz: Sequence[float],
    end_xyz: Sequence[float],
    normal: Sequence[float],
    start_pose: Sequence[float],
    corner_pose: Sequence[float],
    end_pose: Sequence[float],
    max_step_mm: float,
) -> None:
    start_vec = _sub(start_xyz, center)
    end_vec = _sub(end_xyz, center)
    radius = _norm(start_vec)
    if radius < 1e-9:
        _sample_line_segment(out, start_pose, end_pose, max_step_mm)
        return
    signed = math.atan2(_dot(normal, _cross(start_vec, end_vec)), _dot(start_vec, end_vec))
    arc_len = abs(signed) * radius
    steps = max(1, int(math.ceil(arc_len / max(0.5, float(max_step_mm)))))
    rot_start = _euler_to_matrix_deg(start_pose[3], start_pose[4], start_pose[5])
    rot_mid = _euler_to_matrix_deg(corner_pose[3], corner_pose[4], corner_pose[5])
    rot_end = _euler_to_matrix_deg(end_pose[3], end_pose[4], end_pose[5])
    for idx in range(1, steps + 1):
        ratio = idx / float(steps)
        xyz = _add(center, _rotate_vector(start_vec, normal, signed * ratio))
        if ratio <= 0.5:
            rot = _interpolate_rotation_matrix(rot_start, rot_mid, ratio * 2.0)
        else:
            rot = _interpolate_rotation_matrix(rot_mid, rot_end, (ratio - 0.5) * 2.0)
        _append_pose_unique(out, _pose_with_xyz_and_rot(xyz, rot))


def _build_blended_geometry(
    points: Sequence[Sequence[float]],
    *,
    mode: str,
    fillet_mm: float,
    zone_mm: float,
    corner_turn_min_deg: float,
    corner_safety: float,
    min_straight_mm: float,
) -> tuple[list[list[float]], str]:
    if len(points) < 3:
        return [[float(value) for value in point[:6]] for point in points], "linear"

    if mode == "fillet":
        profile = compute_fillet_profile(
            points,
            blend_radius_mm=float(fillet_mm),
            corner_turn_min_deg=float(corner_turn_min_deg),
            radius_safety=float(corner_safety),
            min_straight_mm=float(min_straight_mm),
        )
        d_corner = profile.d_corner
        tan_half = profile.tan_half
        corner_ok = profile.corner_ok
        r_corner = profile.r_corner
    else:
        profile = compute_zone_profile(
            points,
            zone_radius_mm=float(zone_mm),
            corner_turn_min_deg=float(corner_turn_min_deg),
            radius_safety=float(corner_safety),
            min_straight_mm=float(min_straight_mm),
        )
        d_corner = profile.d_corner
        tan_half = profile.tan_half
        corner_ok = profile.corner_ok
        r_corner = [0.0] * len(points)
        for idx in range(1, len(points) - 1):
            if corner_ok[idx] and abs(tan_half[idx]) > 1e-9:
                r_corner[idx] = float(d_corner[idx]) / float(tan_half[idx])

    has_effective_corner = any(corner_ok[idx] and d_corner[idx] > 1e-6 for idx in range(1, len(points) - 1))
    if not has_effective_corner:
        return [[float(value) for value in point[:6]] for point in points], "linear"

    reference_step = 5.0
    out = [[float(value) for value in points[0][:6]]]
    current_pose = [float(value) for value in points[0][:6]]

    for corner_idx in range(1, len(points) - 1):
        corner_pose = [float(value) for value in points[corner_idx][:6]]
        next_pose = [float(value) for value in points[corner_idx + 1][:6]]
        d = float(d_corner[corner_idx])
        if not corner_ok[corner_idx] or d <= 1e-6:
            _sample_line_segment(out, current_pose, corner_pose, reference_step)
            current_pose = corner_pose
            continue

        prev_pose = [float(value) for value in points[corner_idx - 1][:6]]
        dir_in = _unit(_sub(corner_pose[:3], prev_pose[:3]))
        dir_out = _unit(_sub(next_pose[:3], corner_pose[:3]))
        len_in = _dist_xyz(prev_pose, corner_pose)
        len_out = _dist_xyz(corner_pose, next_pose)
        if len_in < 1e-9 or len_out < 1e-9:
            _sample_line_segment(out, current_pose, corner_pose, reference_step)
            current_pose = corner_pose
            continue

        start_xyz = _add(corner_pose[:3], _mul(dir_in, -d))
        end_xyz = _add(corner_pose[:3], _mul(dir_out, d))
        start_pose = _interpolate_pose(prev_pose, corner_pose, _clamp((len_in - d) / len_in, 0.0, 1.0))
        end_pose = _interpolate_pose(corner_pose, next_pose, _clamp(d / len_out, 0.0, 1.0))

        radius = float(r_corner[corner_idx])
        bisector = _unit(_add(_unit(_sub(prev_pose[:3], corner_pose[:3])), _unit(_sub(next_pose[:3], corner_pose[:3]))))
        sin_half = math.sin(math.atan(float(tan_half[corner_idx]))) if abs(tan_half[corner_idx]) > 1e-9 else 0.0
        if radius <= 1e-6 or _norm(bisector) < 1e-9 or abs(sin_half) < 1e-9:
            _sample_line_segment(out, current_pose, start_pose, reference_step)
            _sample_line_segment(out, start_pose, end_pose, reference_step)
            current_pose = end_pose
            continue

        center = _add(corner_pose[:3], _mul(bisector, radius / sin_half))
        start_vec = _sub(start_xyz, center)
        end_vec = _sub(end_xyz, center)
        normal = _unit(_cross(start_vec, end_vec))
        if _norm(normal) < 1e-9:
            _sample_line_segment(out, current_pose, start_pose, reference_step)
            _sample_line_segment(out, start_pose, end_pose, reference_step)
            current_pose = end_pose
            continue

        _sample_line_segment(out, current_pose, start_pose, reference_step)
        _sample_arc_segment(
            out,
            center=center,
            start_xyz=start_xyz,
            end_xyz=end_xyz,
            normal=normal,
            start_pose=start_pose,
            corner_pose=corner_pose,
            end_pose=end_pose,
            max_step_mm=reference_step,
        )
        current_pose = end_pose

    _sample_line_segment(out, current_pose, points[-1], reference_step)
    return out, mode


def _build_reference_geometry(
    points: Sequence[Sequence[float]],
    *,
    fillet_mm: float,
    zone_mm: float,
    corner: CornerMode,
    corner_turn_min_deg: float,
    corner_safety: float,
    min_straight_mm: float,
) -> tuple[list[list[float]], str]:
    normalized = _normalize_points(points)
    mode = str(corner)
    if mode == "linear":
        return normalized, "linear"
    if mode == "zone":
        if float(zone_mm) > 0.0:
            return _build_blended_geometry(
                normalized,
                mode="zone",
                fillet_mm=float(fillet_mm),
                zone_mm=float(zone_mm),
                corner_turn_min_deg=float(corner_turn_min_deg),
                corner_safety=float(corner_safety),
                min_straight_mm=float(min_straight_mm),
            )
        return normalized, "linear"
    if mode == "fillet":
        if float(fillet_mm) > 0.0:
            return _build_blended_geometry(
                normalized,
                mode="fillet",
                fillet_mm=float(fillet_mm),
                zone_mm=float(zone_mm),
                corner_turn_min_deg=float(corner_turn_min_deg),
                corner_safety=float(corner_safety),
                min_straight_mm=float(min_straight_mm),
            )
        return normalized, "linear"

    # auto: keep original priority (zone first, then fillet).
    if float(zone_mm) > 0.0:
        return _build_blended_geometry(
            normalized,
            mode="zone",
            fillet_mm=float(fillet_mm),
            zone_mm=float(zone_mm),
            corner_turn_min_deg=float(corner_turn_min_deg),
            corner_safety=float(corner_safety),
            min_straight_mm=float(min_straight_mm),
        )
    if float(fillet_mm) > 0.0:
        return _build_blended_geometry(
            normalized,
            mode="fillet",
            fillet_mm=float(fillet_mm),
            zone_mm=float(zone_mm),
            corner_turn_min_deg=float(corner_turn_min_deg),
            corner_safety=float(corner_safety),
            min_straight_mm=float(min_straight_mm),
        )
    return normalized, "linear"


def build_trajectory_plan(
    points: Sequence[Sequence[float]],
    *,
    cmdt_s: float,
    fallback_seg_time_s: float = 0.0,
    fillet_radius_mm: float = 0.0,
    zone_radius_mm: float = 0.0,
    blend_mode: CornerMode = "auto",
    min_turn_deg: float = 5.0,
    corner_radius_safety: float = 0.9,
    min_straight_len_mm: float = 0.0,
    speed_mm_s: float | None = None,
    min_z_mm: float | None = None,
    max_target_points: int = 8000,
    max_ori_step_deg: float = 0.25,
    max_centripetal_accel_mm_s2: float = 1000.0,
    centripetal_accel_safety: float = 0.9,
    max_tangential_accel_mm_s2: float = 800.0,
    endpoint_speed_mm_s: float = 0.0,
) -> TrajectoryPlan:
    requested_cmdt = float(cmdt_s)
    if requested_cmdt <= 0.0:
        raise ValueError("cmdt_s must be > 0")
    point_cap = max(1, int(max_target_points))

    geometry_points, geometry_mode = _build_reference_geometry(
        points,
        fillet_mm=float(fillet_radius_mm),
        zone_mm=float(zone_radius_mm),
        corner=blend_mode,
        corner_turn_min_deg=float(min_turn_deg),
        corner_safety=float(corner_radius_safety),
        min_straight_mm=float(min_straight_len_mm),
    )
    if len(geometry_points) < 2:
        raise ValueError("参考几何路径生成失败")

    axis = _build_axis(geometry_points)
    total_axis = axis[-1]
    reference_length = _polyline_length(geometry_points)
    speed_mm_s_est = _estimate_speed_mm_s(
        geometry_points,
        cmdt_s=requested_cmdt,
        seg_time_s=float(fallback_seg_time_s),
        speed_mm_s=speed_mm_s,
    )
    if speed_mm_s_est <= 1e-9:
        speed_mm_s_est = 1.0
    x_path, y_path, z_path = _build_position_paths(geometry_points, axis)
    rotations = [_euler_to_matrix_deg(point[3], point[4], point[5]) for point in geometry_points]

    # Iterate used_cmdt because orientation step limit is per cmdT and affects speed limits.
    used_cmdt = float(requested_cmdt)
    if point_cap < 2:
        point_cap = 2

    dense_axis: list[float] = []
    dense_xyz: list[list[float]] = []
    dense_rotations: list[list[list[float]]] = []
    dense_arc: list[float] = []
    total_arc = 0.0
    targets: list[list[float]] = []

    for _ in range(6):
        target_step_mm = max(0.1, speed_mm_s_est * used_cmdt)
        dense_axis = _dense_axis_values(total_axis, target_step_mm, len(geometry_points))
        dense_xyz = [
            _evaluate_position(axis_value, x_path=x_path, y_path=y_path, z_path=z_path, min_z=min_z_mm)
            for axis_value in dense_axis
        ]
        dense_rotations = [_evaluate_rotation(axis_value, axis, rotations) for axis_value in dense_axis]
        dense_arc = _build_arc_table(dense_xyz)
        total_arc = float(dense_arc[-1]) if dense_arc else 0.0

        if total_arc <= 1e-9:
            last_pose = _pose_with_xyz_and_rot(dense_xyz[-1], dense_rotations[-1])
            return TrajectoryPlan(
                geometry_mode=geometry_mode,
                geometry_points=geometry_points,
                targets=[last_pose],
                reference_length_mm=reference_length,
                target_length_mm=0.0,
                requested_cmdt_s=requested_cmdt,
                used_cmdt_s=used_cmdt,
                max_target_points=point_cap,
                max_ori_step_deg=float(max_ori_step_deg),
            )

        # Build speed limits along arc-length samples.
        n = len(dense_arc)
        kappa = [0.0] * n
        for i in range(1, n - 1):
            kappa[i] = _triangle_curvature_mm_inv(dense_xyz[i - 1], dense_xyz[i], dense_xyz[i + 1])
        if n >= 2:
            kappa[0] = kappa[1]
            kappa[-1] = kappa[-2]

        dtheta_ds = [0.0] * n
        for i in range(1, n - 1):
            ds = float(dense_arc[i + 1]) - float(dense_arc[i - 1])
            if ds <= 1e-9:
                continue
            dtheta = _rotation_angle_between_mats_deg(dense_rotations[i - 1], dense_rotations[i + 1])
            dtheta_ds[i] = float(dtheta) / ds
        if n >= 2:
            dtheta_ds[0] = dtheta_ds[1]
            dtheta_ds[-1] = dtheta_ds[-2]

        v_max = [float(speed_mm_s_est)] * n
        a_c = float(max_centripetal_accel_mm_s2) * _clamp(float(centripetal_accel_safety), 0.05, 1.0)
        if a_c > 0.0:
            for i in range(n):
                if kappa[i] > 1e-12:
                    v_max[i] = min(v_max[i], math.sqrt(max(0.0, a_c / kappa[i])))

        # Orientation step limit is defined as max deg per cmdT.
        rot_step = max(0.05, float(max_ori_step_deg))
        omega_max = rot_step / max(1e-6, float(used_cmdt))  # deg/s
        for i in range(n):
            if dtheta_ds[i] > 1e-12:
                v_max[i] = min(v_max[i], omega_max / dtheta_ds[i])

        # Enforce tangential acceleration limits and endpoint speeds.
        v = _enforce_accel_limits(
            dense_arc,
            v_max,
            a_t_max=float(max_tangential_accel_mm_s2) if float(max_tangential_accel_mm_s2) > 0.0 else 1e9,
            v_start=float(endpoint_speed_mm_s),
            v_end=float(endpoint_speed_mm_s),
        )
        t_table = _build_time_table(dense_arc, v)
        total_time = float(t_table[-1]) if t_table else 0.0
        if total_time <= 1e-9:
            total_time = used_cmdt

        # Cap point count by increasing cmdT if needed.
        required_points = int(math.ceil(total_time / float(used_cmdt))) + 1
        if required_points > point_cap:
            new_cmdt = max(float(used_cmdt), total_time / float(point_cap - 1))
            if new_cmdt > float(used_cmdt) + 1e-9:
                used_cmdt = new_cmdt
                continue

        # Generate uniform-time targets.
        sample_count = max(2, required_points)
        targets = []
        for k in range(sample_count):
            tq = min(float(total_time), float(k) * float(used_cmdt))
            sq = _invert_monotone_table(dense_arc, t_table, tq)
            axis_q = _find_axis_from_arc(sq, dense_arc, dense_axis)
            xyz = _evaluate_position(axis_q, x_path=x_path, y_path=y_path, z_path=z_path, min_z=min_z_mm)
            rot = _evaluate_rotation(axis_q, axis, rotations)
            targets.append(_pose_with_xyz_and_rot(xyz, rot))

        break

    if not targets:
        last_pose = geometry_points[-1][:]
        if min_z_mm is not None:
            last_pose[2] = max(float(min_z_mm), last_pose[2])
        targets = [last_pose]
    else:
        last_pose = geometry_points[-1][:]
        if min_z_mm is not None:
            last_pose[2] = max(float(min_z_mm), last_pose[2])
        if _dist_xyz(targets[-1], last_pose) > 1e-6 or _rotation_angle_deg(targets[-1], last_pose) > 1e-4:
            targets.append(last_pose)

    target_length = _polyline_length([geometry_points[0]] + targets)
    return TrajectoryPlan(
        geometry_mode=geometry_mode,
        geometry_points=geometry_points,
        targets=targets,
        reference_length_mm=reference_length,
        target_length_mm=target_length,
        requested_cmdt_s=requested_cmdt,
        used_cmdt_s=used_cmdt,
        max_target_points=point_cap,
        max_ori_step_deg=float(max_ori_step_deg),
    )


def build_targets(
    points: Sequence[Sequence[float]],
    *,
    cmdt_s: float,
    fallback_seg_time_s: float = 0.0,
    fillet_radius_mm: float = 0.0,
    zone_radius_mm: float = 0.0,
    blend_mode: CornerMode = "auto",
    min_turn_deg: float = 5.0,
    corner_radius_safety: float = 0.9,
    min_straight_len_mm: float = 0.0,
    speed_mm_s: float | None = None,
    min_z_mm: float | None = None,
    max_target_points: int = 8000,
    max_ori_step_deg: float = 0.25,
    max_centripetal_accel_mm_s2: float = 1000.0,
    centripetal_accel_safety: float = 0.9,
    max_tangential_accel_mm_s2: float = 800.0,
    endpoint_speed_mm_s: float = 0.0,
) -> list[list[float]]:
    return build_trajectory_plan(
        points,
        cmdt_s=cmdt_s,
        fallback_seg_time_s=fallback_seg_time_s,
        fillet_radius_mm=fillet_radius_mm,
        zone_radius_mm=zone_radius_mm,
        blend_mode=blend_mode,
        min_turn_deg=min_turn_deg,
        corner_radius_safety=corner_radius_safety,
        min_straight_len_mm=min_straight_len_mm,
        speed_mm_s=speed_mm_s,
        min_z_mm=min_z_mm,
        max_target_points=max_target_points,
        max_ori_step_deg=max_ori_step_deg,
        max_centripetal_accel_mm_s2=max_centripetal_accel_mm_s2,
        centripetal_accel_safety=centripetal_accel_safety,
        max_tangential_accel_mm_s2=max_tangential_accel_mm_s2,
        endpoint_speed_mm_s=endpoint_speed_mm_s,
    ).targets

