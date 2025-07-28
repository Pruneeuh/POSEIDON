import numpy as np
from numpy import ndarray


def generate_points_3D() -> ndarray:
    """Generate random 3D points in a space 4*4*4
    points_3D (np.ndarray): Random 3D points of shape (4, 3).
    """

    return np.random.rand(4, 3) * [50.0, 50.0, 50.0]  # 4 points in 3D (4*3)


def projection_points_2D(points_3D: ndarray, C: ndarray, R: ndarray, A: ndarray) -> ndarray:
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


def compute_features_vectors(points3D: ndarray, C: ndarray, R: ndarray) -> ndarray:
    """
    This function computes the features vectors for P3P algorithm.
    Args:
        points3D (np.ndarray): array with the 4 3D points = [ P1, P2, P3 ] (3*3)
        C (np.ndarray): camera position matrix : (1*3)
        R (np.ndarray): camera rotation matrix : (3*3)
    Returns:
        featuresVect (np.ndarray): array with the features vectors (3*3)
    """

    P1: ndarray = np.reshape(points3D[0], (3, 1))
    P2: ndarray = np.reshape(points3D[1], (3, 1))
    P3: ndarray = np.reshape(points3D[2], (3, 1))

    C = np.reshape(C, (3, 1))  # (3*1)

    v1: ndarray = R @ (P1 - C)  # (3*1)
    v2: ndarray = R @ (P2 - C)
    v3: ndarray = R @ (P3 - C)

    f1: ndarray = v1 / np.linalg.norm(v1)
    f2: ndarray = v2 / np.linalg.norm(v2)
    f3: ndarray = v3 / np.linalg.norm(v3)

    f1 = np.reshape(f1 / np.linalg.norm(f1), (1, 3))
    f2 = np.reshape(f2 / np.linalg.norm(f2), (1, 3))
    f3 = np.reshape(f3 / np.linalg.norm(f3), (1, 3))

    featuresVect: ndarray = np.concatenate((f1, f2, f3), axis=0)

    return featuresVect  # Return the features vectors need in P3P (one row = one feature vector)
