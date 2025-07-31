import numpy as np 
from numpy import ndarray

from poseidon.torch.utils import get_feature_vectors, convert_matrix_numpy_to_batch

def get_feature_vector_np(points2D : ndarray, A : ndarray ) -> ndarray : 
    """
    Compute feature vectors from 2D points and intrinsic matrix.

    Args:
        points2D (np.ndarray): 2D points in image coordinates (3, 2).
        A (np.ndarray): Camera intrinsic matrix (3, 3).

    Returns:
        featuresVect (np.ndarray): feature vectors for each point (3, 3).
    """
    # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
    ones = np.ones((3,1))
    p_h = np.concatenate([points2D, ones], axis=-1)  # (3, 3)

    # Inverse of intrinsic matrix
    A_inv = np.linalg.inv(A)  # (3, 3)

    # Transform all points at once: A_inv @ p_h^T → (batch_size, 3, 3)
    featuresVect = A_inv @ p_h.T  # (3, 3)

    norms = np.linalg.norm(featuresVect, axis=0, keepdims=True)  # (1, 3)
    featuresVect = featuresVect / norms  # Normalize each vector
    featuresVect = featuresVect.T  # (3, 3)

    return featuresVect  # (3, 3)

A = np.array([[1000, 0, 320],
              [0, 1000, 240],
              [0, 0, 1]])

points2D = np.array([[ 400.9729, -326.3301],
         [ 400.9146, -326.2762],
         [ 401.8035, -326.3407]])

featuresVect = get_feature_vector_np(points2D, A)

A_torch = convert_matrix_numpy_to_batch(A)
points2D_torch = convert_matrix_numpy_to_batch(points2D)
featuresVect_torch = get_feature_vectors(points2D_torch, A_torch)

print("Features Vectors (Numpy):\n", featuresVect)
print("Features Vectors (Torch):\n", featuresVect_torch)
