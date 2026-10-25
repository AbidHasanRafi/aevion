"""ROS 2 message construction and the rclpy publisher, against a stand-in for rclpy (no ROS installation needed)."""

import math
import sys
import types

import numpy as np
import pytest

from aevion.bridges import ros2
from aevion.bridges.ros2 import Ros2Bridge, Ros2Hook, clock_msg, fill_message, imu_msg, odometry_msg, pointcloud2_msg, stamp, tf_msg
from aevion.core import rotation as rot
from aevion.core.backend import NumpyBackend
from aevion.engine import Simulation

B = NumpyBackend()


# ------------------------------------------------------------------------------------------------ a stand-in for rclpy
class FakeMsg:
    """Any attribute is a nested message until assigned - enough to receive what ``fill_message`` writes."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        child = FakeMsg()
        object.__setattr__(self, name, child)
        return child


def install_fake_ros(monkeypatch, inited=False):
    state = types.SimpleNamespace(inited=inited, published=[], nodes=[], shutdowns=0)

    class Node:
        def __init__(self):
            self.destroyed = False

        def create_publisher(self, cls, topic, depth):
            return types.SimpleNamespace(publish=lambda m: state.published.append((topic, cls.__name__, m)))

        def destroy_node(self):
            self.destroyed = True

    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: state.inited
    rclpy.init = lambda: setattr(state, "inited", True)

    def create_node(name):
        n = Node()
        n.name = name
        state.nodes.append(n)
        return n

    def shutdown():
        state.inited = False
        state.shutdowns += 1

    rclpy.create_node, rclpy.shutdown = create_node, shutdown
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    for mod, names in {
        "nav_msgs.msg": ["Odometry"],
        "sensor_msgs.msg": ["Imu", "PointCloud2", "PointField"],
        "rosgraph_msgs.msg": ["Clock"],
        "tf2_msgs.msg": ["TFMessage"],
        "geometry_msgs.msg": ["TransformStamped"],
    }.items():
        m = types.ModuleType(mod)
        for n in names:
            setattr(m, n, type(n, (FakeMsg,), {}))
        monkeypatch.setitem(sys.modules, mod, m)
    return state


# ------------------------------------------------------------------------------------------------ message content
def test_time_stamps_split_seconds_and_nanoseconds_without_overflow():
    assert stamp(12.25) == dict(sec=12, nanosec=250_000_000)
    assert stamp(0.0) == dict(sec=0, nanosec=0)
    assert stamp(3.9999999999) == dict(sec=4, nanosec=0)  # rounding must carry into the seconds, never emit nanosec = 1e9
    assert clock_msg(1.5) == dict(clock=dict(sec=1, nanosec=500_000_000))


def test_odometry_has_a_world_pose_with_xyzw_quaternion_and_a_body_frame_twist():
    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # 90 degrees about +z: body x points along world +y
    m = odometry_msg(2.0, [1.0, 2.0, 3.0], [0.0, 5.0, 0.0], q, [0.1, 0.2, 0.3])
    assert m["header"] == dict(stamp=dict(sec=2, nanosec=0), frame_id="odom") and m["child_frame_id"] == "base_link"
    assert m["pose"]["pose"]["position"] == dict(x=1.0, y=2.0, z=3.0)
    o = m["pose"]["pose"]["orientation"]
    assert (o["x"], o["y"], o["z"], o["w"]) == pytest.approx((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    lin = m["twist"]["twist"]["linear"]  # ENU velocity (0, 5, 0) seen from that body is straight ahead: (5, 0, 0)
    assert (lin["x"], lin["y"], lin["z"]) == pytest.approx((5.0, 0.0, 0.0), abs=1e-12)
    assert m["twist"]["twist"]["angular"] == dict(x=0.1, y=0.2, z=0.3)
    assert len(m["pose"]["covariance"]) == 36 and m["pose"]["covariance"][0] == pytest.approx(0.05**2) and m["pose"]["covariance"][35] > 0


def test_odometry_velocity_matches_the_library_rotation_for_arbitrary_attitudes():
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    v = rng.normal(size=3)
    lin = odometry_msg(0.0, [0, 0, 0], v, q, [0, 0, 0])["twist"]["twist"]["linear"]
    assert np.allclose([lin["x"], lin["y"], lin["z"]], rot.rotate_inv(B, q[None, :], v[None, :])[0])


def test_imu_message_carries_specific_force_and_marks_missing_orientation():
    m = imu_msg(1.0, [0.0, 0.0, 9.81], [0.0, 0.1, 0.0])
    assert m["linear_acceleration"] == dict(x=0.0, y=0.0, z=9.81) and m["angular_velocity"]["y"] == 0.1
    assert m["orientation_covariance"][0] == -1.0  # ROS convention: no orientation estimate
    with_q = imu_msg(1.0, [0, 0, 9.81], [0, 0, 0], quat=[1.0, 0.0, 0.0, 0.0])
    assert with_q["orientation_covariance"][0] == 0.0 and with_q["orientation"]["w"] == 1.0


def test_pointcloud_packs_little_endian_float32_and_drops_non_finite_points():
    pts = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, np.inf, 0.0]])
    m = pointcloud2_msg(0.5, pts)
    assert m["width"] == 2 and m["height"] == 1 and m["point_step"] == 12 and m["row_step"] == 24 and len(m["data"]) == 24
    assert [f["name"] for f in m["fields"]] == ["x", "y", "z"] and [f["offset"] for f in m["fields"]] == [0, 4, 8]
    assert np.array_equal(np.frombuffer(m["data"], "<f4").reshape(-1, 3), [[1, 2, 3], [4, 5, 6]])
    with_i = pointcloud2_msg(0.5, pts[[0, 2]], intensity=[0.25, 0.75])
    assert with_i["point_step"] == 16 and [f["name"] for f in with_i["fields"]][-1] == "intensity"
    assert np.frombuffer(with_i["data"], "<f4").reshape(-1, 4)[1, 3] == 0.75
    assert pointcloud2_msg(0.0, np.zeros((0, 3)))["width"] == 0


def test_tf_message_holds_one_transform_between_the_two_frames():
    m = tf_msg(1.0, [1, 2, 3], [0.5, 0.5, 0.5, 0.5])
    (t,) = m["transforms"]
    assert (t["header"]["frame_id"], t["child_frame_id"]) == ("odom", "base_link")
    assert t["transform"]["translation"] == dict(x=1.0, y=2.0, z=3.0) and t["transform"]["rotation"] == dict(x=0.5, y=0.5, z=0.5, w=0.5)


# ------------------------------------------------------------------------------------------------ rclpy publisher
def test_constructing_the_bridge_without_rclpy_explains_what_to_do(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)  # makes `import rclpy` raise ImportError
    with pytest.raises(ImportError, match="rclpy"):
        Ros2Bridge()


def test_fill_message_copies_nested_fields_and_lists_of_messages(monkeypatch):
    install_fake_ros(monkeypatch)
    msg = fill_message(sys.modules["nav_msgs.msg"].Odometry(), odometry_msg(1.0, [1, 2, 3], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0]))
    assert msg.pose.pose.position.x == 1.0 and msg.header.frame_id == "odom" and msg.header.stamp.sec == 1 and len(msg.pose.covariance) == 36
    tf = fill_message(sys.modules["tf2_msgs.msg"].TFMessage(), tf_msg(1.0, [1, 2, 3], [1, 0, 0, 0]))
    assert isinstance(tf.transforms[0], sys.modules["geometry_msgs.msg"].TransformStamped) and tf.transforms[0].transform.translation.z == 3.0
    pc = fill_message(sys.modules["sensor_msgs.msg"].PointCloud2(), pointcloud2_msg(0.0, [[1, 2, 3]]))
    assert isinstance(pc.fields[0], sys.modules["sensor_msgs.msg"].PointField) and pc.fields[1].name == "y" and len(pc.data) == 12


def test_the_bridge_initialises_rclpy_once_publishes_on_configurable_topics_and_shuts_down(monkeypatch):
    st = install_fake_ros(monkeypatch)
    br = Ros2Bridge("sim", topics={"odom": "/uav/odom"})
    assert st.inited and st.nodes[0].name == "sim"
    br.publish_state(1.0, [1, 2, 3], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], accel=[0, 0, 9.81])
    br.publish_state(1.1, [1, 2, 3], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0])
    topics = [(t, c) for t, c, _ in st.published]
    assert topics == [
        ("clock", "Clock"),
        ("/uav/odom", "Odometry"),
        ("tf", "TFMessage"),
        ("imu", "Imu"),
        ("clock", "Clock"),
        ("/uav/odom", "Odometry"),
        ("tf", "TFMessage"),
    ]
    assert len(br.pubs) == 4  # publishers are created lazily and reused
    br.close()
    assert st.nodes[0].destroyed and st.shutdowns == 1 and not st.inited


def test_an_already_initialised_rclpy_is_left_alone_until_close(monkeypatch):
    st = install_fake_ros(monkeypatch, inited=True)
    br = Ros2Bridge()
    br.close()
    assert st.shutdowns == 1  # close() shuts rclpy down only because it is running


def test_the_hook_publishes_the_simulated_vehicle_at_the_requested_rate(monkeypatch):
    st = install_fake_ros(monkeypatch)
    br = Ros2Bridge()
    hook = Ros2Hook(br, every=50)
    arch = Simulation({"duration": 1.0, "trajectory": {"type": "hover", "params": {"pos": [0, 0, 2.0]}}}, hooks=[hook]).run()
    odom = [m for t, c, m in st.published if c == "Odometry"]
    assert 4 <= len(odom) <= 12 and st.shutdowns == 1  # on_end closed the bridge
    last = odom[-1]
    assert last.pose.pose.position.z == pytest.approx(arch["pos"][-1, 0, 2], abs=0.05)
    assert [c for _, c, _ in st.published if c == "Imu"]  # the accelerometer channel is published too
    assert ros2.Ros2Hook(br, every=0).every == 1
