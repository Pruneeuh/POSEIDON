import numpy as np
from numpy import ndarray


def generate_points_3D() -> ndarray:
    """Generate random 3D points in a space 4*4*4
    points_3D (np.ndarray): Random 3D points of shape (4, 3).
    """

    return np.random.rand(4, 3) * [2.0, 2.0, 2.0]  # 4 points in 3D (4*3)


def projection_points_2D(
    points_3D: ndarray, C: ndarray, R: ndarray, A: ndarray
) -> ndarray:
    """Project 3D points to 2D using camera parameters.
    Args:
        points_3D (np.ndarray): 4 3D points of shape (4, 3).
        C (np.ndarray): Camera position vector of shape (3, 1).
        R (np.ndarray): Camera rotation matrix of shape (3, 3).
        A (np.ndarray): Intrinsic camera matrix of shape (3, 3).
    Returns:
        point2D (np.ndarray): Projected 2D points of shape (4,2).
    """

    PI: ndarray = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)  # (3*4)

    Rt: ndarray = np.concatenate((R, C), axis=1)  # (3*4)
    Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)  # (4*4)

    points_2D: ndarray = np.zeros((4, 2))  # Initialize the array for 2D points (4*2)
    # Project each 3D point to 2D
    for i in range(4):
        point3D_bis: ndarray = np.concatenate(
            (np.reshape(points_3D[i, :], (3, 1)), np.array([[1]])), axis=0
        )  # (4*1)
        point2D: ndarray = A @ PI @ Rt @ point3D_bis  # 2D point = [u, v, w] (3*1)
        point2D = point2D / point2D[2]  # 2D point = [u, v, 1] (3*1)
        points_2D[i, :] = point2D[:2].reshape(1, 2)  # Store the 2D point in the array

    return points_2D
