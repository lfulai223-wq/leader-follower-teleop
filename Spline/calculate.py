"""计算工具：向量/距离/交融半径上限等。

约定：位姿 pose 为 [x, y, z, rx, ry, rz]（单位与 Fairino SDK 一致）。
本文件不包含任何机器人通讯/运动指令，只做纯计算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Pose = List[float]


@dataclass
class FilletProfile:
	"""圆角交融（fillet）在每个拐角的半径/切点切走长度配置。

	- r_corner[i]：拐角 i 的实际半径（已考虑 blend_radius_mm、几何上限、safety、相邻不重合缩放）
	- d_corner[i]：在相邻直线段上“切走”的长度 d=r*tan(theta/2)
	- tan_half[i]：tan(theta/2)
	- theta[i]：拐角角度 theta（弧度）
	- r_geom_max[i]：仅按几何约束得到的 r_max（未乘 safety/未考虑 overlap）
	"""

	r_corner: List[float]
	d_corner: List[float]
	tan_half: List[float]
	theta: List[float]
	r_geom_max: List[float]
	corner_ok: List[bool]
	overlap_scaled: List[bool]


@dataclass
class ZoneProfile:
	"""交融区（zone）在每个拐角的“切走长度”配置。

	这里的 zone_radius_mm 语义按你确认的定义：
	- 每个拐点 B 有一个以 B 为圆心、半径 R 的交融区
	- 沿相邻两条直线方向，从 B 各退让 d=R，得到切点 P/Q
	- 若相邻拐点在同一段上发生重合，则对各自的 d(=R) 做比例缩小以满足不回拉约束

	字段含义：
	- d_corner[i]：拐角 i 的实际切走长度 d（即 R_i^eff），已考虑全局上限、几何上限(L1/L2)、safety、相邻不重合缩放
	- theta[i] / tan_half[i]：拐角角度与 tan(theta/2)，用于把 d 转换为圆弧半径 r=d/tan_half
	- corner_ok[i]：该点是否为有效拐角（非共线/非掉头/非极短段）
	"""

	d_corner: List[float]
	tan_half: List[float]
	theta: List[float]
	corner_ok: List[bool]
	overlap_scaled: List[bool]


def compute_fillet_profile(
	points: Sequence[Sequence[float]],
	*,
	blend_radius_mm: float,
	corner_turn_min_deg: float = 5.0,
	radius_safety: float = 0.9,
	min_straight_mm: float = 0.0,
	max_iter: int = 12,
) -> FilletProfile:
	"""计算当前参数下的 fillet 配置（不生成 targets，仅做纯几何/约束缩放）。

	相邻不重合约束正对应你说的：
	- 在中间那段直线（例如 2->3）上，上一个拐角切点为 q，下一个拐角切点为 m
	- 要求 q 不在 m 的右侧（不回拉）：等价于 d[2] + d[3] <= |P2P3| - min_straight_mm
	- 当 min_straight_mm=0 时，极限情况是 q 与 m 重合（中间直线长度变 0）
	"""
	n = len(points)
	r_corner = [0.0] * n
	d_corner = [0.0] * n
	tan_half = [0.0] * n
	theta_list = [0.0] * n
	r_geom_max = [0.0] * n
	corner_ok = [False] * n
	overlap_scaled = [False] * n

	# 初始：每个拐角按几何上限 + safety + 全局上限 blend_radius_mm 得到 r_i
	for i in range(1, n - 1):
		A = points[i - 1]
		B = points[i]
		C = points[i + 1]
		v1 = v_sub(A[:3], B[:3])
		v2 = v_sub(C[:3], B[:3])
		L1 = norm(v1)
		L2 = norm(v2)
		if L1 < 1e-6 or L2 < 1e-6:
			continue
		u1 = unit(v1)
		u2 = unit(v2)
		cos_th = clamp(dot(u1, u2), -1.0, 1.0)
		theta = math.acos(cos_th)
		# 近似掉头：不做交融
		if theta < 1e-3:
			continue
		# 近似共线：转角过小不做交融（转角=|pi-theta|）
		turn = abs(math.pi - theta)
		if turn < math.radians(float(corner_turn_min_deg)):
			continue
		t = math.tan(theta / 2.0)
		if abs(t) < 1e-6:
			continue
		r_max = min(L1, L2) / t
		r_geom_max[i] = float(r_max)
		tan_half[i] = float(t)
		theta_list[i] = float(theta)
		r = min(float(blend_radius_mm), float(r_max) * float(radius_safety))
		if r < 1e-6:
			continue
		r_corner[i] = float(r)
		d_corner[i] = float(r) * float(t)
		corner_ok[i] = True

	# 相邻圆角不重合（不回拉）约束：必要时缩放相邻两个半径（d 与 r 线性相关）
	for _ in range(int(max_iter)):
		changed = False
		for i in range(n - 1):
			seg_len = dist_xyz(points[i], points[i + 1])
			max_cut = max(0.0, seg_len - float(min_straight_mm))
			sum_d = d_corner[i] + d_corner[i + 1]
			if sum_d > max_cut + 1e-6 and sum_d > 1e-9:
				s = max_cut / sum_d
				if 1 <= i <= n - 2 and corner_ok[i]:
					r_corner[i] *= s
					d_corner[i] *= s
					overlap_scaled[i] = True
				if 1 <= (i + 1) <= n - 2 and corner_ok[i + 1]:
					r_corner[i + 1] *= s
					d_corner[i + 1] *= s
					overlap_scaled[i + 1] = True
				changed = True
		if not changed:
			break

	return FilletProfile(
		r_corner=r_corner,
		d_corner=d_corner,
		tan_half=tan_half,
		theta=theta_list,
		r_geom_max=r_geom_max,
		corner_ok=corner_ok,
		overlap_scaled=overlap_scaled,
	)


def compute_zone_profile(
	points: Sequence[Sequence[float]],
	*,
	zone_radius_mm: float,
	corner_turn_min_deg: float = 5.0,
	radius_safety: float = 0.9,
	min_straight_mm: float = 0.0,
	max_iter: int = 12,
) -> ZoneProfile:
	"""计算当前参数下的 zone 配置（不生成 targets，仅做纯几何/约束缩放）。

	对每个拐角 i：初始取 d_i = min(zone_radius_mm, |P_{i-1}P_i|, |P_iP_{i+1}|) * safety。
	对每段 Pi->P{i+1}：相邻不重合（不回拉）约束为
		d[i] + d[i+1] <= |PiPi+1| - min_straight_mm
	若超限，按比例缩小相邻两端的 d。
	"""
	n = len(points)
	d_corner = [0.0] * n
	tan_half = [0.0] * n
	theta_list = [0.0] * n
	corner_ok = [False] * n
	overlap_scaled = [False] * n

	R = float(zone_radius_mm)
	if n < 3 or R <= 0.0:
		return ZoneProfile(
			d_corner=d_corner,
			tan_half=tan_half,
			theta=theta_list,
			corner_ok=corner_ok,
			overlap_scaled=overlap_scaled,
		)

	# 初始：每个拐角按全局上限 R + 单点几何上限(L1/L2) + safety 得到 d_i
	for i in range(1, n - 1):
		A = points[i - 1]
		B = points[i]
		C = points[i + 1]
		v1 = v_sub(A[:3], B[:3])
		v2 = v_sub(C[:3], B[:3])
		L1 = norm(v1)
		L2 = norm(v2)
		if L1 < 1e-6 or L2 < 1e-6:
			continue
		u1 = unit(v1)
		u2 = unit(v2)
		cos_th = clamp(dot(u1, u2), -1.0, 1.0)
		theta = math.acos(cos_th)
		# 近似掉头：不做交融
		if theta < 1e-3:
			continue
		# 近似共线：转角过小不做交融（转角=|pi-theta|）
		turn = abs(math.pi - theta)
		if turn < math.radians(float(corner_turn_min_deg)):
			continue
		t = math.tan(theta / 2.0)
		if abs(t) < 1e-9:
			continue
		tan_half[i] = float(t)
		theta_list[i] = float(theta)
		d = min(R, float(L1), float(L2)) * float(radius_safety)
		if d < 1e-6:
			continue
		d_corner[i] = float(d)
		corner_ok[i] = True

	# 相邻不重合（不回拉）约束：必要时缩放相邻两个 d
	for _ in range(int(max_iter)):
		changed = False
		for i in range(n - 1):
			seg_len = dist_xyz(points[i], points[i + 1])
			max_cut = max(0.0, seg_len - float(min_straight_mm))
			sum_d = d_corner[i] + d_corner[i + 1]
			if sum_d > max_cut + 1e-6 and sum_d > 1e-9:
				s = max_cut / sum_d
				if 1 <= i <= n - 2 and corner_ok[i]:
					d_corner[i] *= s
					overlap_scaled[i] = True
				if 1 <= (i + 1) <= n - 2 and corner_ok[i + 1]:
					d_corner[i + 1] *= s
					overlap_scaled[i + 1] = True
				changed = True
		if not changed:
			break

	return ZoneProfile(
		d_corner=d_corner,
		tan_half=tan_half,
		theta=theta_list,
		corner_ok=corner_ok,
		overlap_scaled=overlap_scaled,
	)


def lerp_pose(a: Sequence[float], b: Sequence[float], t: float) -> Pose:
	"""位姿 6 维线性插值（XYZ + RxRyRz 都做线性）。"""
	return [float(x + (y - x) * t) for x, y in zip(a, b)]


def clamp(x: float, lo: float, hi: float) -> float:
	return lo if x < lo else hi if x > hi else x


def v_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def v_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
	return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def v_mul(a: Sequence[float], s: float) -> List[float]:
	return [a[0] * s, a[1] * s, a[2] * s]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
	return [
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	]


def norm(a: Sequence[float]) -> float:
	return math.sqrt(dot(a, a))


def unit(a: Sequence[float], eps: float = 1e-9) -> List[float]:
	n = norm(a)
	if n < eps:
		return [0.0, 0.0, 0.0]
	return [a[0] / n, a[1] / n, a[2] / n]


def rotate(v: Sequence[float], axis_unit: Sequence[float], angle_rad: float) -> List[float]:
	"""Rodrigues 旋转公式。axis_unit 必须是单位向量。"""
	k = axis_unit
	cos_a = math.cos(angle_rad)
	sin_a = math.sin(angle_rad)
	term1 = v_mul(v, cos_a)
	term2 = v_mul(cross(k, v), sin_a)
	term3 = v_mul(k, dot(k, v) * (1.0 - cos_a))
	return v_add(v_add(term1, term2), term3)


def pose_with_xyz(pose: Sequence[float], xyz: Sequence[float]) -> Pose:
	"""保留姿态(rx,ry,rz)不变，只替换 xyz。"""
	return [xyz[0], xyz[1], xyz[2], float(pose[3]), float(pose[4]), float(pose[5])]


def dist_xyz(p: Sequence[float], q: Sequence[float]) -> float:
	dx = p[0] - q[0]
	dy = p[1] - q[1]
	dz = p[2] - q[2]
	return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_blend_radius_limits(points: Sequence[Sequence[float]]) -> List[Tuple[int, float, float, float, float]]:
	"""计算每个拐角的最大交融半径（mm）。

	对拐角 B（A->B->C），设两段长度为 L1=|BA|, L2=|BC|，夹角为 θ：
	切点距离 d = r*tan(θ/2)，要求 d<=min(L1,L2)
	=> r_max = min(L1,L2)/tan(θ/2)
	"""
	limits: List[Tuple[int, float, float, float, float]] = []
	for i in range(1, len(points) - 1):
		A = points[i - 1]
		B = points[i]
		C = points[i + 1]

		v1 = v_sub(A[:3], B[:3])
		v2 = v_sub(C[:3], B[:3])
		L1 = norm(v1)
		L2 = norm(v2)
		if L1 < 1e-6 or L2 < 1e-6:
			continue

		u1 = unit(v1)
		u2 = unit(v2)
		cos_th = clamp(dot(u1, u2), -1.0, 1.0)
		theta = math.acos(cos_th)
		# 近似共线/掉头，不做交融；上限视作无意义，跳过
		if theta < 1e-3 or abs(math.pi - theta) < 1e-3:
			continue

		tan_half = math.tan(theta / 2.0)
		if abs(tan_half) < 1e-6:
			continue

		r_max = min(L1, L2) / tan_half
		limits.append((i, float(r_max), float(theta), float(L1), float(L2)))
	return limits


def compute_zone_radius_limits(points: Sequence[Sequence[float]]) -> List[Tuple[int, float, float, float, float]]:
	"""计算每个拐角的最大交融区半径 R 上限（mm）。

	按 zone 语义，R 直接对应切走长度 d，需要满足 d<=min(L1,L2)。
	因此单拐角几何上限：R_max = min(|BA|, |BC|)。

	返回 (corner_idx, R_max, theta, L1, L2)。
	"""
	limits: List[Tuple[int, float, float, float, float]] = []
	for i in range(1, len(points) - 1):
		A = points[i - 1]
		B = points[i]
		C = points[i + 1]

		v1 = v_sub(A[:3], B[:3])
		v2 = v_sub(C[:3], B[:3])
		L1 = norm(v1)
		L2 = norm(v2)
		if L1 < 1e-6 or L2 < 1e-6:
			continue
		u1 = unit(v1)
		u2 = unit(v2)
		cos_th = clamp(dot(u1, u2), -1.0, 1.0)
		theta = math.acos(cos_th)
		# 近似共线/掉头，不做交融；上限视作无意义，跳过
		if theta < 1e-3 or abs(math.pi - theta) < 1e-3:
			continue

		R_max = min(float(L1), float(L2))
		limits.append((i, float(R_max), float(theta), float(L1), float(L2)))
	return limits


def recommend_zone_radius(
	points: Sequence[Sequence[float]],
	*,
	radius_safety: float = 0.9,
	min_leg_mm: float = 1.0,
) -> Tuple[float, int]:
	"""推荐一个“对有效拐角”的交融区半径（忽略极短段导致的假性极小上限）。

	返回 (recommended_zone_mm, limiting_corner_idx)。
	注意：这里推荐的是单拐角几何上限（R<=min(L1,L2)），相邻重合会在 profile 中自动缩放。
	"""
	limits = compute_zone_radius_limits(points)
	if not limits:
		return 0.0, -1
	meaningful = [
		(i, R_max, theta, L1, L2)
		for (i, R_max, theta, L1, L2) in limits
		if min(float(L1), float(L2)) >= float(min_leg_mm)
	]
	pick = meaningful if meaningful else limits
	min_i, R_min, _, _, _ = min(pick, key=lambda x: x[1])
	return float(R_min) * float(radius_safety), int(min_i)


def compute_global_radius_upper(points: Sequence[Sequence[float]], *, min_straight_mm: float = 0.0) -> float:
	"""估算“全局 BLEND_RADIUS_MM”的上限（raw，不含 safety）。

	考虑两类约束：
	1) 单个拐角几何上限 r_max
	2) 相邻拐角不重合：对每段 Pi->Pi+1，要求 d[i]+d[i+1] <= |PiPi+1|-min_straight_mm
	   若用同一全局 r，且 d[i]=r*tan_half[i]，则可得
	   r <= (|seg|-min_straight_mm)/(tan_half[i]+tan_half[i+1])
	"""
	limits = compute_blend_radius_limits(points)
	upper = float("inf")
	for _, r_max, _, _, _ in limits:
		upper = min(upper, float(r_max))

	# 计算每个拐角的 tan(theta/2)
	n = len(points)
	tan_half: List[float] = [0.0] * n
	for i in range(1, n - 1):
		A = points[i - 1]
		B = points[i]
		C = points[i + 1]
		v1 = v_sub(A[:3], B[:3])
		v2 = v_sub(C[:3], B[:3])
		if norm(v1) < 1e-6 or norm(v2) < 1e-6:
			continue
		u1 = unit(v1)
		u2 = unit(v2)
		cos_th = clamp(dot(u1, u2), -1.0, 1.0)
		theta = math.acos(cos_th)
		if theta < 1e-3 or abs(math.pi - theta) < 1e-3:
			continue
		t = math.tan(theta / 2.0)
		if abs(t) < 1e-6:
			continue
		tan_half[i] = float(t)

	for i in range(n - 1):
		seg_len = dist_xyz(points[i], points[i + 1])
		max_cut = max(0.0, seg_len - float(min_straight_mm))
		den = tan_half[i] + tan_half[i + 1]
		if den > 1e-9:
			upper = min(upper, max_cut / den)

	return float(upper)

