import torch
from torch import Tensor


def get_feature_vectors(points2D: Tensor, A: Tensor) -> Tensor:
    """
    Compute feature vectors from 2D points and intrinsic matrix.

    Args:
        points2D (torch.Tensor): 2D points in image coordinates (batch_size, 3, 2).
        A (torch.Tensor): Camera intrinsic matrix (batch_size, 3, 3).

    Returns:
        featuresVect (torch.tensor): feature vectors for each point (batch_size, 3, 3).
    """
    batch_size = points2D.shape[0]

    # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
    ones = torch.ones((batch_size, 3, 1), dtype=points2D.dtype, device=points2D.device)
    p_h = torch.cat([points2D, ones], dim=-1)  # (batch_size, 3, 3)

    # Inverse of intrinsic matrix
    A_inv = torch.linalg.inv(A)  # (batch_size, 3, 3)

    # Transform all points at once: A_inv @ p_h^T → (batch_size, 3, 3)
    featuresVect = torch.matmul(A_inv, p_h.transpose(1, 2))  # (batch_size, 3, 3)

    # Normalize each vector (along rows in dim=1, for each column)
    norms = torch.norm(featuresVect, dim=1, keepdim=True)  # (batch_size, 1, 3)
    featuresVect = featuresVect / norms
    featuresVect = featuresVect.transpose(1, 2)  # (batch_size, 3, 3)

    return featuresVect  # (batch_size, 3, 3)


def generate_synthetic_2D3Dpoints(R, C, A, points3D):
    """
    Generate synthetic corresponding 2D and 3D points for P3P problem.
    Args:
        R (torch.Tensor): Rotation matrix (batch_size,3,3).
        C (torch.Tensor): Camera center (batch_size,3).
        A (torch.Tensor): Camera intrinsic matrix (batch_size,3,3).
        points3D (torch.Tensor): 3D points in world coordinates (batch_size,4,3).
    Returns:
        points2D (torch.Tensor): Projected 2D points in image coordinates (batch_size,4,2).
    """
    batch_size = R.shape[0]  # Get the batch size from the first dimension of R
    points3D = torch.transpose(points3D, 1, 2)
    print("points3D shape:", points3D)  # (batch_size, 3, 4   )

    # Compute camera translation vector from rotation R and position C
    t = torch.matmul(-R, torch.reshape(C, (batch_size, 3, 1)))  # (batch_size, 3, 1)

    # Build projection matrix: P = A [R|t]
    Rt = torch.cat([R, t], dim=2)  # (batch_size, 3, 4)

    P = torch.matmul(A, Rt)  # (batch_size, 3, 4)

    # Convert 3D points to homogeneous coordinates (4x3)
    points3D_h = torch.cat(
        [points3D, torch.ones(batch_size, 1, 4, dtype=torch.float64)], dim=1
    )  # (batch_size, 4, 4)

    # Project 3D points to 2D image plane using projection matrix
    proj = torch.matmul(P, points3D_h)  # (batch_size, 3, 3)

    proj = proj / proj[:, 2, :]  # normalize homogeneous coordinates

    # Extract 2D image coordinates (3 points, shape 3x2)
    points2D = proj[:, :2, :].transpose(1, 2)  # (batch_size, 3, 2)

    return points2D
