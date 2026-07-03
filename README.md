# leader-follower-teleop

Real-time **leader–follower teleoperation** for robot arms: hand-drag a passive
leader arm and a 6-DoF follower arm mirrors your motion in real time — with a
multi-stage smoothing pipeline that makes the follower **smooth, responsive, and
safe** instead of jittery.

Built for an **Alicia-D** leader arm driving a **FAIRINO FR3** follower over
`ServoJ` at 100–125 Hz, but the control/smoothing logic is hardware-agnostic and
easy to port.

> Use case: collecting high-quality demonstration data for **imitation learning /
> behavior cloning**, or general master–slave remote manipulation.

---

## ✨ Why this repo

Naively piping leader joint angles to the follower produces high-frequency
jitter, unsafe velocity spikes, and stop-go stutter. The core of this project is
a **cascaded smoothing pipeline** that fixes each of those, one stage at a time:

```
leader raw angles
   │
   ├─▶ [1] OneEuro adaptive low-pass   ← strong shake rejection at rest,
   │                                      opens up during motion (stays responsive)
   ├─▶ [2] fixed low-pass              ← hard cut of hand tremor band, speed-independent
   │
   ├─▶ [3] axis mapping + joint limits ← map to follower, clamp to safe range
   │
   ├─▶ [4] critically-damped spring    ← natural accel/decel, kills start-stop jerk
   │        (+ acceleration & velocity caps for safety)
   │
   └─▶ ServoJ @ 100Hz ─▶ follower arm
```

Each stage targets a specific failure mode — see [How it works](#-how-it-works).

## Features

- **Two-stage adaptive + fixed filtering** — reject tremor without sacrificing follow-through.
- **Critically-damped spring tracking** — smooth velocity profiles, no overshoot, no stutter.
- **Acceleration & velocity limits** — bounded, predictable, safe follower motion.
- **Relative mode by default** — no zero-calibration needed; follower stays put until you move.
- **Configurable axis mapping** — per-joint order and sign for any leader→follower pairing.
- **Robust connection + servo-error auto-recovery** — retries on connect, clears alarms and
  restarts servo mode on `ServoJ` faults.
- **All tunables in one place** — a single `CONFIG` block at the top of the script.
- **Data recording tool** — `record.py` captures single trajectories or multi-episode datasets to CSV.
- **Simulation mode** — `--sim` runs against a mock follower with a live matplotlib view (no hardware needed).

## Requirements

- Python 3.10+
- `numpy`
- **Leader SDK** — `alicia_d_sdk` (for the Alicia-D arm), installed separately.
- **Follower SDK** — the FAIRINO Python SDK (`fairino`), obtained from the robot vendor and
  placed under `Spline/fairino390/` (not included here — proprietary).

> Both vendor SDKs are **not** bundled in this repo. Swap in your own arm SDKs by adapting the
> thin driver calls in `alicia_teleop_fr3.py` (`create_robot`, `ServoJ`, `GetActualJointPosDegree`, …).

## Quick start

```bash
# 1. Dry run — no follower, just print mapped target angles (verify axis mapping)
python alicia_teleop_fr3.py --no-robot

# 2. Simulation — mock follower + live plot, no hardware
python alicia_teleop_fr3.py --sim

# 3. Real follower (teach pendant in AUTO mode, robot enabled)
python alicia_teleop_fr3.py --robot-ip 192.168.57.2
```

Recording demonstrations:

```bash
# single trajectory
python record.py single --output trajectories/demo_001.csv

# multi-episode dataset
python record.py dataset --session pick_and_place --num_episodes 20 --episode_duration 8
```

## Configuration

All tunables live in the `CONFIG` block at the top of `alicia_teleop_fr3.py`
(every value is also overridable via `--flag`):

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `RATE` | Control frequency (Hz) | `100` |
| `FILTER_MINCUTOFF` | OneEuro cutoff (Hz) — lower = steadier, slower | `2.0` |
| `FILTER_BETA` | OneEuro speed adaptivity — higher = more responsive | `0.05` |
| `TREMOR_CUTOFF` | 2nd-stage fixed low-pass cutoff (Hz) | `3.0` |
| `SPRING_OMEGA` | Spring natural freq ωₙ (rad/s) — higher = snappier | `12.0` |
| `MAX_ACCEL` | Joint acceleration cap (°/s²) | `500` |
| `MAX_VEL` | Joint velocity cap (°/s) | `60` |
| `DEAD_ZONE` | Min change (°) before a command is sent | `0.03` |
| `MIN_ANGLE` / `MAX_ANGLE` | Per-joint safety limits (°) | see file |
| `AXIS_ORDER` / `AXIS_SIGN` | Leader→follower joint mapping | see file |

## 🔬 How it works

1. **OneEuro adaptive filter** — cutoff rises with speed (`cutoff = mincutoff + beta·|ẋ|`):
   heavy smoothing when the hand is still, light smoothing when moving fast. Solves the
   *steadiness vs. responsiveness* trade-off.
2. **Fixed low-pass** — OneEuro's speed-adaptivity lets fast tremor leak through; a second
   fixed-cutoff stage hard-caps the tremor band regardless of speed.
3. **Axis mapping + limits** — remap leader joints to follower joints (configurable order/sign)
   and clamp into each joint's safe range, avoiding controller limit alarms at the source.
4. **Critically-damped spring (ζ=1)** — `acc = ωₙ²·(target − pos) − 2ωₙ·vel`, integrated at the
   control rate. Produces continuous accel/decel curves: micro-pauses of the hand decay
   smoothly instead of hard-stopping, and re-starts ramp up without jerk. Acceleration and
   velocity caps act as safety bounds on top.

## Safety notes

- The follower moves in real time — keep the workspace clear and a stop within reach.
- Start with `--no-robot` / `--sim` to validate axis mapping before touching hardware.
- Tighten `MIN_ANGLE` / `MAX_ANGLE`, `MAX_VEL`, and `MAX_ACCEL` for your cell before running.

## License

[MIT](LICENSE)
