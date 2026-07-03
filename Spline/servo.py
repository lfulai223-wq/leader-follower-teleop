"""Fairino ServoCart 平滑轨迹示例（SDK 调用版）。

这个文件是最终运行入口：
- 点位直接以 Python list 形式传入（不读文件）。
- 回 HOME / 去 HOME 等业务动作只放在这里；`servo_cart_sdk.py` 保持纯粹通用。
"""

from __future__ import annotations

from servo_cart_sdk import ServoCartConfig, connect_robot, plan_from_points, run_servo_spline
import time
HOME_DESC_POS: list[float] = [381.19598388671875, -73.28435516357422, 561.1694946289062, 157.77699279785156, 15.183841705322266, 124.45417022705078]
HOME_JOINT_POS: list[float] = [4.701240539550781, -69.14685821533203, -101.00321960449219, -73.18351745605469, 88.69316864013672, 146.93832397460938]


def go_home(robot, *, tool: int, user: int, vel: float = 10.0) -> None:
	ret = robot.MoveJ(
		joint_pos=list(HOME_JOINT_POS),
		desc_pos=list(HOME_DESC_POS),
		vel=float(vel),
		tool=int(tool),
		user=int(user),
	)
	if int(ret) != 0:
		raise RuntimeError(f"回 HOME 失败: {ret}")


def main() -> None:
	# ===== 你常用的可调参数（直接改这里即可） =====
	robot_ip = "192.168.57.2"
	tool = 0
	user = 0
	# 走的点：直接写成 Python list（每个点为 [x,y,z,rx,ry,rz]）
	points: list[list[float]] = [
		[381.19598388671875, -73.28435516357422, 561.1694946289062, 157.77699279785156, 15.183841705322266, 124.45417022705078],
		[434.04791259765625, -16.104280471801758, 498.05438232421875, 172.00469970703125, 8.618888854980469, 142.35311889648438],
		[591.4530639648438, 125.79022216796875, 420.72119140625, -179.9586944580078, 0.45278432965278625, 138.4640655517578],
		[591.4521484375, 125.79798126220703, 236.62872314453125, -179.95848083496094, 0.4526742696762085, 138.46458435058594]
	]

	cmdt_s = 0.004
	speed_mm_s = 600.0
	max_ori_step_deg = 10

	# 其它常用开关（需要时再打开/修改）
	swap_rpy_order = False
	auto_swap_rpy_order = False
	verbose = True

	robot = connect_robot(robot_ip)
	try:
		go_home(robot, tool=tool, user=user)

		cfg = ServoCartConfig(
			robot_ip=str(robot_ip),
			tool=int(tool),
			user=int(user),
			points=[list(map(float, p[:6])) for p in points],
			swap_rpy_order=bool(swap_rpy_order),
			auto_swap_rpy_order=bool(auto_swap_rpy_order),
			cmdt_s=float(cmdt_s),
			speed_mm_s=float(speed_mm_s),
			max_ori_step_deg=float(max_ori_step_deg),
			verbose=bool(verbose),
		)
		plan, targets = plan_from_points(robot, cfg)
 
		# MoveCart 到样条起点（入口负责，SDK 里不做）
		move_to_start_err = int(
			robot.MoveCart(
				desc_pos=list(plan.geometry_points[0]),
				vel=50,
				tool=int(tool),
				user=int(user),
			)
		)
		if move_to_start_err != 0:
			raise RuntimeError(f"MoveCart 到起点失败: {move_to_start_err}")
		start = time.time()
		err = run_servo_spline(
			robot,
			targets,
			cmdt_s=float(plan.used_cmdt_s),
			queue_low_watermark=int(cfg.queue_low_watermark),
			queue_high_watermark=int(cfg.queue_high_watermark),
			queue_guard=int(cfg.queue_guard),
			queue_prefill_target=int(cfg.queue_prefill_target),
			queue_poll_period_s=float(cfg.queue_poll_period_s),
			log_queue_len=bool(cfg.log_queue_len),
		)
		print(f"运行结束: errcode={int(err)}")
		print(f"总耗时: {time.time() - start:.3f} 秒")
		# go_home(robot, tool=tool, user=user)
	finally:
		try:
			robot.CloseRPC()
		except Exception:
			pass


if __name__ == "__main__":
	main()
