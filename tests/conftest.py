import numpy as np
import pytest

from autolabel.schemas import EgoMotion


def synth_ego(profile: str = "brake", dt: float = 0.1, n: int = 200, seed: int = 0) -> EgoMotion:
    """Synthetic 20 s ego-motion.

    brake     : 12 m/s, brakes to 0 at t=8..11 s, waits, accelerates at t=14 s
    turn      : 8 m/s constant, 90 deg left turn between t=8 and t=12 s
    cruise    : constant speed, straight
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    speed = np.full(n, 12.0)
    curv = np.zeros(n)
    if profile == "brake":
        for i, ti in enumerate(t):
            if 8 <= ti < 11:
                speed[i] = 12.0 * (1 - (ti - 8) / 3)
            elif 11 <= ti < 14:
                speed[i] = 0.0
            elif ti >= 14:
                speed[i] = min(12.0, 2.5 * (ti - 14))
    elif profile == "turn":
        speed[:] = 8.0
        yaw_rate = np.where((t >= 8) & (t < 12), np.pi / 2 / 4, 0.0)   # rad/s
        curv = yaw_rate / speed
    elif profile == "cruise":
        pass
    speed = np.clip(speed + rng.normal(0, 0.05, n), 0, None)
    ax = np.gradient(speed, dt)
    heading = np.cumsum(curv * speed * dt)
    x = np.cumsum(speed * np.cos(heading) * dt)
    y = np.cumsum(speed * np.sin(heading) * dt)
    return EgoMotion(
        timestamps_us=(t * 1e6).astype(np.int64),
        positions=np.stack([x, y, np.zeros(n)], 1),
        velocities=np.stack([speed, np.zeros(n), np.zeros(n)], 1),
        accelerations=np.stack([ax, np.zeros(n), np.zeros(n)], 1),
        curvatures=curv,
        source="sensor",
    )


@pytest.fixture
def ego_brake():
    return synth_ego("brake")


@pytest.fixture
def ego_turn():
    return synth_ego("turn")


@pytest.fixture
def ego_cruise():
    return synth_ego("cruise")
