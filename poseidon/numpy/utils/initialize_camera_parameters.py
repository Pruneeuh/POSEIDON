import numpy as np
from numpy import ndarray


def generate_camera_parameters() -> ndarray:
    """Generate random camera parameters.
    Returns:
        A (np.ndarray): Intrinsic camera matrix of shape (3, 3).
    """
    # Definition of the camera parameters
    # focal length
    fx: float = np.random.uniform(500, 3000)
    fy: float = fx * np.random.uniform(0.95, 1.05)
    # center
    cx: float = np.random.uniform(300, 1000)
    cy: float = np.random.uniform(200, 800)

    A: ndarray = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    )  # intraseca matrix of the camera (3*3)
    return A


def generate_rotation_matrix_with_angles(yaw, pitch, rax) -> ndarray:
    """Generate rotation matrix from yaw, pitch, and rax.
    The matrix need to be orthogonal and have a determinant of 1.

    Returns:
        R (np.ndarray) : Rotation matrix of shape (3, 3).
    """
    Rx: ndarray = np.array(
        [[1, 0, 0], [0, np.cos(rax), -np.sin(rax)], [0, np.sin(rax), np.cos(rax)]]
    )
    Ry: ndarray = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz: ndarray = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    R: ndarray = Rz @ Ry @ Rx  # rotation matrix  (3*3)

    return R


def generate_rotation_matrix() -> ndarray:
    """Generate a random rotation matrix.
    The angles are randomly generated in the pages given by the LARD dataset
    The matrix need to be orthogonal and have a determinant of 1.

    Returns:
        R (np.ndarray) : Rotation matrix of shape (3, 3).
    """
    yaw: float = np.radians(np.random.uniform(-10, 10))
    pitch: float = np.radians(np.random.uniform(-8, 8))
    rax: float = np.radians(np.random.uniform(-10, 10))

    return generate_rotation_matrix_with_angles(yaw, pitch, rax)


def generate_position_matrix() -> ndarray:
    """Generate a random position vector.
    Returns:
        C (np.ndarray): Position vector of shape (3, 1).
    """
    x: float = np.random.uniform(-50, 50)
    y: float = np.random.uniform(-50, 50)
    z: float = np.random.uniform(1, 50)
    return np.array([[x], [y], [z]])  # position vector (3*1)
