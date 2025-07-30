import torch
from torch import Tensor


def get_feature_vectors(points2D: Tensor, A: Tensor) -> Tensor:
    """
    Compute feature vectors from 2D points and intrinsic matrix.

    Args:
        points2D (torch.Tensor): 2D points in image coordinates (batch_size,3,2).
        A (torch.Tensor): Camera intrinsic matrix (batch_size,3,3).

    Returns:
        featuresVect (torch.tensor):  feature vectors for each point (batch_size, 3, 3).)
    """
    batch_size: int = points2D.shape[0]
    ones: Tensor = torch.ones((batch_size, 3, 1), dtype=torch.float64)  # (batch_size, 3, 1)

    # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
    p_h: Tensor = torch.cat([points2D, ones], dim=-1)  # (batch_size, 3, 3)

    A_inv: Tensor = torch.linalg.inv(A)  # Inverse of intrinsic matrix

    featuresVectList: list = []

    for i in range(3):
        p_h_i: Tensor = p_h[:, i].unsqueeze(-1)  # (batch_size, 3,1)

        # Apply inverse of intrinsic matrix to get direction vector in camera frame
        fi: Tensor = torch.matmul(A_inv, p_h_i).squeeze(-1)  # (batch_size, 3)

        # Normalize to get a unit vector (bearing direction)
        fi = fi / torch.norm(fi, dim=1, keepdim=True)  # Normalize along the first dimension

        featuresVectList.append(fi)  # Reshape to (batch_size, 3, 1)

    # Stack into a matrix: shape (3, 3) where each column is f1, f2, f3
    featuresVect: Tensor = torch.stack(featuresVectList, dim=1)  # (batch_size, 3, 3)
    return featuresVect


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
