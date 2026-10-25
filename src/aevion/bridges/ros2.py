"""ROS 2 messages from Aevion state, and an optional publisher.

Aevion's frames already follow REP 103 (ENU world, FLU body), so nothing is flipped. Only the quaternion order differs (ROS
``x, y, z, w``; Aevion ``w, x, y, z``), and ROS expresses an odometry twist in the child (body) frame. The functions build plain
dictionaries that mirror ``nav_msgs/Odometry``, ``sensor_msgs/Imu``, ``sensor_msgs/PointCloud2``, ``rosgraph_msgs/Clock`` and
``tf2_msgs/TFMessage``, usable without ROS installed.

:class:`Ros2Bridge` publishes them through ``rclpy`` (imported only when the bridge is constructed). It follows the documented ``rclpy``
API and is tested against a stand-in; it has **not been run against a ROS 2 installation**, so check topic names, QoS and frames on
first use.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..core import rotation as rot
from ..core.backend import NumpyBackend
from ..engine.hooks import Hook

__all__ = ["stamp", "odometry_msg", "imu_msg", "pointcloud2_msg", "clock_msg", "tf_msg", "Ros2Bridge", "Ros2Hook", "fill_message"]

B = NumpyBackend()
FLOAT32 = 7  # sensor_msgs/PointField datatype


def stamp(t: float) -> Dict[str, int]:
    """``builtin_interfaces/Time`` from seconds."""
    total = round(float(t) * 1e9)
    return dict(sec=total // 1_000_000_000, nanosec=total % 1_000_000_000)


def _q_xyzw(q: Sequence[float]) -> Dict[str, float]:
    return dict(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0]))


def _vec(v: Sequence[float]) -> Dict[str, float]:
    return dict(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def odometry_msg(
    t: float,
    pos: Any,
    vel: Any,
    quat: Any,
    omega: Any,
    frame_id: str = "odom",
    child_frame_id: str = "base_link",
    pos_std: float = 0.05,
    vel_std: float = 0.05,
) -> Dict[str, Any]:
    """``nav_msgs/Odometry``: pose in ``frame_id`` (ENU), twist in ``child_frame_id`` (the body), covariances on the diagonal."""
    q = np.asarray(quat, float)[None, :]
    v_body = rot.rotate_inv(B, q, np.asarray(vel, float)[None, :])[0]
    pc = [0.0] * 36
    tc = [0.0] * 36
    for i in range(3):
        pc[7 * i] = pos_std**2
        pc[7 * (i + 3)] = (0.01) ** 2
        tc[7 * i] = vel_std**2
        tc[7 * (i + 3)] = (0.01) ** 2
    return dict(
        header=dict(stamp=stamp(t), frame_id=frame_id),
        child_frame_id=child_frame_id,
        pose=dict(pose=dict(position=_vec(pos), orientation=_q_xyzw(q[0])), covariance=pc),
        twist=dict(twist=dict(linear=_vec(v_body), angular=_vec(omega)), covariance=tc),
    )


def imu_msg(t: float, accel: Any, gyro: Any, quat: Optional[Any] = None, frame_id: str = "base_link") -> Dict[str, Any]:
    """``sensor_msgs/Imu``: specific force (including the reaction to gravity, as ROS expects) and angular velocity in the body frame."""
    unknown = [-1.0] + [0.0] * 8  # covariance[0] = -1 marks a field as not provided
    return dict(
        header=dict(stamp=stamp(t), frame_id=frame_id),
        orientation=_q_xyzw(quat) if quat is not None else dict(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=[0.0] * 9 if quat is not None else unknown,
        angular_velocity=_vec(gyro),
        angular_velocity_covariance=[0.0] * 9,
        linear_acceleration=_vec(accel),
        linear_acceleration_covariance=[0.0] * 9,
    )


def pointcloud2_msg(t: float, points: Any, intensity: Optional[Any] = None, frame_id: str = "base_link") -> Dict[str, Any]:
    """``sensor_msgs/PointCloud2`` of ``(N, 3)`` points (float32 x, y, z and optionally intensity); non-finite points are dropped."""
    p = np.asarray(points, np.float32).reshape(-1, 3)
    cols = [p]
    names = ["x", "y", "z"]
    if intensity is not None:
        cols.append(np.asarray(intensity, np.float32).reshape(-1, 1))
        names.append("intensity")
    arr = np.concatenate(cols, axis=1)
    arr = arr[np.isfinite(arr).all(axis=1)]
    step = 4 * arr.shape[1]
    return dict(
        header=dict(stamp=stamp(t), frame_id=frame_id),
        height=1,
        width=int(arr.shape[0]),
        fields=[dict(__msg__="sensor_msgs.msg.PointField", name=n, offset=4 * i, datatype=FLOAT32, count=1) for i, n in enumerate(names)],
        is_bigendian=False,
        point_step=step,
        row_step=step * int(arr.shape[0]),
        data=arr.astype("<f4").tobytes(),
        is_dense=True,
    )


def clock_msg(t: float) -> Dict[str, Any]:
    return dict(clock=stamp(t))


def tf_msg(t: float, pos: Any, quat: Any, frame_id: str = "odom", child_frame_id: str = "base_link") -> Dict[str, Any]:
    """``tf2_msgs/TFMessage`` with the single transform ``frame_id -> child_frame_id``."""
    return dict(
        transforms=[
            dict(
                __msg__="geometry_msgs.msg.TransformStamped",
                header=dict(stamp=stamp(t), frame_id=frame_id),
                child_frame_id=child_frame_id,
                transform=dict(translation=_vec(pos), rotation=_q_xyzw(quat)),
            )
        ]
    )


def _load(type_name: str) -> Any:
    mod, _, cls = type_name.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


def fill_message(msg: Any, data: Dict[str, Any]) -> Any:
    """Copy a dictionary produced above into a ROS message object (nested messages, arrays, and lists of messages)."""
    for k, v in data.items():
        if k == "__msg__":
            continue
        if isinstance(v, dict):
            fill_message(getattr(msg, k), v)
        elif isinstance(v, list) and v and isinstance(v[0], dict) and "__msg__" in v[0]:
            items = []
            for d in v:
                m = _load(d["__msg__"])()
                fill_message(m, d)
                items.append(m)
            setattr(msg, k, items)
        else:
            setattr(msg, k, v)
    return msg


class Ros2Bridge:
    """Publish Aevion state on ROS 2 topics (``/odom``, ``/imu``, ``/clock``, ``/tf``; point clouds on request)."""

    TYPES = {
        "odom": "nav_msgs.msg.Odometry",
        "imu": "sensor_msgs.msg.Imu",
        "clock": "rosgraph_msgs.msg.Clock",
        "tf": "tf2_msgs.msg.TFMessage",
        "points": "sensor_msgs.msg.PointCloud2",
    }

    def __init__(self, node_name: str = "aevion", topics: Optional[Dict[str, str]] = None, queue_size: int = 10) -> None:
        try:
            self.rclpy = importlib.import_module("rclpy")
        except ImportError as ex:
            raise ImportError(
                "Ros2Bridge needs ROS 2's Python client (rclpy): source your ROS 2 installation before starting Python, then retry"
            ) from ex
        if not self.rclpy.ok():
            self.rclpy.init()
        self.node = self.rclpy.create_node(node_name)
        self.topics = {"odom": "odom", "imu": "imu", "clock": "clock", "tf": "tf", "points": "points"}
        self.topics.update(topics or {})
        self.pubs: Dict[str, Any] = {}
        self.queue_size = queue_size

    def _pub(self, kind: str) -> Any:
        if kind not in self.pubs:
            self.pubs[kind] = self.node.create_publisher(_load(self.TYPES[kind]), self.topics[kind], self.queue_size)
        return self.pubs[kind]

    def publish(self, kind: str, data: Dict[str, Any]) -> None:
        msg = fill_message(_load(self.TYPES[kind])(), data)
        self._pub(kind).publish(msg)

    def publish_state(self, t: float, pos: Any, vel: Any, quat: Any, omega: Any, accel: Optional[Any] = None) -> None:
        self.publish("clock", clock_msg(t))
        self.publish("odom", odometry_msg(t, pos, vel, quat, omega))
        self.publish("tf", tf_msg(t, pos, quat))
        if accel is not None:
            self.publish("imu", imu_msg(t, accel, omega, quat))

    def close(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self.rclpy.ok():
                self.rclpy.shutdown()


class Ros2Hook(Hook):
    """Engine hook that publishes vehicle ``vehicle`` through a :class:`Ros2Bridge` every ``every`` physics steps."""

    name = "ros2"
    priority = 95

    def __init__(self, bridge: Ros2Bridge, vehicle: int = 0, every: int = 5) -> None:
        self.bridge, self.vehicle, self.every = bridge, int(vehicle), max(int(every), 1)

    def post_step(self, sim: Any) -> None:
        if sim.k % self.every:
            return
        B_, s, i = sim.B, sim.state, self.vehicle
        g = lambda x: np.asarray(B_.to_numpy(x))[i]  # noqa: E731
        self.bridge.publish_state(sim.t, g(s.pos), g(s.vel), g(s.quat), g(s.omega), g(s.f_body))

    def on_end(self, sim: Any, archive: Any = None) -> None:
        self.bridge.close()
