"""Pose conventions.

AirSim camera frame: x forward, y right, z down (world = NED).
Exported poses use the OpenCV camera frame (x right, y down, z forward), camera-to-world, world kept in NED.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# columns: OpenCV axes expressed in the AirSim camera frame
CV_TO_AIRSIM_CAM = np.array([[0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0]], dtype=float)


def quat_to_R(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_xyzw(R):
    return Rotation.from_matrix(R).as_quat()


def camera_c2w_cv(cam_pose):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(cam_pose.orientation) @ CV_TO_AIRSIM_CAM
    T[:3, 3] = [cam_pose.position.x_val, cam_pose.position.y_val, cam_pose.position.z_val]
    return T


def intrinsics(width, height, fov_deg):
    fx = width / 2.0 / np.tan(np.radians(fov_deg) / 2.0)
    return {"width": width, "height": height, "fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0}
