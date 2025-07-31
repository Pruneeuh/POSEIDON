import numpy as np
from numpy import ndarray


def get_feature_vector(points2D: ndarray, A: ndarray) -> ndarray:
    """
    Compute feature vectors from 2D points and intrinsic matrix.

    Args:
        points2D (np.ndarray): 2D points in image coordinates (3, 2).
        A (np.ndarray): Camera intrinsic matrix (3, 3).

    Returns:
        featuresVect (np.ndarray): feature vectors for each point (3, 3).
    """
    # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
    ones: ndarray = np.ones((3, 1))
    p_h: ndarray = np.concatenate([points2D, ones], axis=-1)  # (3, 3)

    # Inverse of intrinsic matrix
    A_inv: ndarray = np.linalg.inv(A)  # (3, 3)

    # Transform all points at once: A_inv @ p_h^T → (batch_size, 3, 3)
    featuresVect: ndarray = A_inv @ p_h.T  # (3, 3)

    norms: ndarray = np.linalg.norm(featuresVect, axis=0, keepdims=True)  # (1, 3)
    featuresVect = featuresVect / norms  # Normalize each vector
    featuresVect = featuresVect.T  # (3, 3)

    return featuresVect  # (3, 3)


def generate_synthetic_2D3Dpoints(R: ndarray, C: ndarray, A: ndarray, points3D: ndarray) -> ndarray:
    """
    Generate synthetic corresponding 2D and 3D points for P3P problem.
    Args:
        R (np.ndarray): Rotation matrix (3,3).
        C (np.ndarray): Camera center (3,).
        A (np.ndarray): Camera intrinsic matrix (3,3).
        points3D (np.ndarray): 3D points in world coordinates (4,3).
    Returns:
        points2D (np.ndarray): Projected 2D points in image coordinates (4,2).
    """
    points3D = points3D.T  # (3, 4)

    # Compute camera translation vector from rotation R and position C
    t: ndarray = -R @ C.reshape(3, 1)  # (3, 1)

    # Build projection matrix: P = A [R|t]
    Rt: ndarray = np.concatenate([R, t], axis=1)  # (3, 4)

    P: ndarray = A @ Rt  # (3, 4)

    # Convert 3D points to homogeneous coordinates (4x3)
    points3D_h: ndarray = np.concatenate([points3D, np.ones((1, 4))], axis=0)  # (4, 4)

    # Project 3D points to 2D image plane using projection matrix
    proj: ndarray = P @ points3D_h  # (3, 4)

    proj = proj / proj[2, :]  # normalize homogeneous coordinates

    # Extract 2D image coordinates (4 points, shape 4x2)
    points2D: ndarray = proj[:2, :].T  # (4, 2)

    return points2D
