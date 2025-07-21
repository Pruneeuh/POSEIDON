import torch
from numpy import ndarray
from torch import Tensor


def convert_matrix_numpy_to_batch(X_numpy: ndarray) -> Tensor:
    """Convert a numpy  matrix to a torch tensor with batch_size = 1
    Args:
        X_numpy (np.ndarray)
    Returns:
        X_torch (torch.Tensor)
    """
    return torch.tensor(X_numpy, dtype=torch.float64).unsqueeze(0)  # Convert to torch tensor


def generate_points_3D(batch_size) -> Tensor:
    """
    Generate random 3D points in a space 4*4*4
    Args:
        batch_size (int): Number of points to generate.
    Returns:
        points_3D (torch.Tensor): Random 3D points of shape (batch_size, 4, 3).
    """
    return torch.rand(batch_size, 3, 4) * 4 - 2


def compute_features_vectors(points_3D: Tensor, C: Tensor, R: Tensor) -> Tensor:
    """
    This function computes the features vectors for P3P algorithm.
    Args:
        points3D : tensor with the 3D points : (batch_size,3,4)
        C : tensor with the camera position : (batch_size,3,1)
        R : tensor with the camera rotation matrix : (batch_size,3,3)
    Returns:
        featuresVect : tensor with the features vectors (batch_size,3,3)
    """
    batch_size: int = points_3D.shape[0]  # Get the batch size from the first dimension of points_3D

    P1: Tensor = torch.reshape(points_3D[:, 0], (batch_size, 3, 1))  # (batch_size, 3, 1)
    P2: Tensor = torch.reshape(points_3D[:, 1], (batch_size, 3, 1))
    P3: Tensor = torch.reshape(points_3D[:, 2], (batch_size, 3, 1))

    C = torch.reshape(C, (batch_size, 3, 1))  # (batch_size, 3, 1)

    v1: Tensor = torch.matmul(R, (P1 - C))  # (batch_size, 3, 1)
    v2: Tensor = torch.matmul(R, (P2 - C))  # (batch_size, 3, 1)
    v3: Tensor = torch.matmul(R, (P3 - C))  # (batch_size, 3, 1)

    f1: Tensor = v1 / torch.norm(v1, dim=1, keepdim=True)  # (batch_size, 3, 1)
    f2: Tensor = v2 / torch.norm(v2, dim=1, keepdim=True)  # (batch_size, 3, 1)
    f3: Tensor = v3 / torch.norm(v3, dim=1, keepdim=True)  # (batch_size, 3, 1)

    f1 = torch.reshape(f1, (batch_size, 1, 3))  # (batch_size,1,3)
    f2 = torch.reshape(f2, (batch_size, 1, 3))
    f3 = torch.reshape(f3, (batch_size, 1, 3))

    featuresVect: Tensor = torch.cat((f1, f2, f3), dim=1)  # (batch_size, 3, 3)

    return featuresVect  # Return the features vectors need in P3P


def projection_one_point3D_to2D(point3D: Tensor, C: Tensor, R: Tensor, A: Tensor) -> Tensor:
    """
    This function projects 3D point to 2D point using the camera parameters.
    Args:
        point3D : tensor with the 3D points : (batch_size,3,1)
        C : tensor with the camera position : (batch_size,3,1)
        R : tensor with the camera rotation matrix : (batch_size,3,3)
        A : tensor with the intrinsic camera matrix : (batch_size,3,3)
    Returns:
        point2D : tensor with the 2D points : (batch_size,2,1)
    """

    batch_size: int = point3D.shape[0]  # Get the batch size from the first dimension of point3D

    PI: Tensor = torch.cat(
        (torch.eye(3, dtype=torch.float64), torch.zeros((3, 1), dtype=torch.float64)),
        dim=1,
    )  # (3,4)
    PI_batch: Tensor = PI.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size,3,4)

    Rt: Tensor = torch.cat((R, C), dim=2)  # (batch_size,3,4)
    Rt = torch.cat(
        (
            Rt,
            (torch.tensor([[0, 0, 0, 1]], dtype=torch.float64)).repeat(batch_size, 1, 1),
        ),
        dim=1,
    )  # (batch_size,4,4)

    point3D_bis: Tensor = torch.cat(
        (
            torch.reshape(point3D, (batch_size, 3, 1)),
            (torch.tensor([[1]], dtype=torch.float64)).repeat(batch_size, 1, 1),
        ),
        dim=1,
    )  # (batch_size,4,1)
    point2D: Tensor = torch.matmul(
        A, torch.matmul(PI_batch, torch.matmul(Rt, point3D_bis))
    )  # (batch_size,3,1)

    # point2D = torch.tensordot(torch.tensordot(torch.tensordot(A,PI,dims=1),Rt,dims=1),point3D_bis,dims=1)  # 2D point = [u, v, w] (3*1)
    point2D = point2D / point2D[:, 2:3, :]  # 2D point = [u, v, 1] (batch_size,3,1)

    return point2D[:, :2]


def projection_all_point3D_to2D(points3D: Tensor, C: Tensor, R: Tensor, A: Tensor) -> Tensor:
    """
    This function projects all 3D points to 2D points using the camera parameters.
    Args:
        points3D : tensor with the 3D points : (batch_size,4,3)
        C : tensor with the camera position : (batch_size,3,1)
        R : tensor with the camera rotation matrix : (batch_size,3,3)
        A : tensor with the intrinsic camera matrix : (batch_size,3,3)
    Returns:
        points2D : tensor with the 2D points : (batch_size,4,2)
    """

    batch_size: int = points3D.shape[0]  # Get the batch size from the first dimension of points3D

    points2D: Tensor = torch.empty(
        batch_size, 4, 2, dtype=torch.float64
    )  # Initialize tensor for 2D points
    for i in range(4):
        Pi: Tensor = points3D[:, i, :].unsqueeze(-1)
        pi: Tensor = projection_one_point3D_to2D(Pi, C, R, A)
        points2D[:, i, :] = pi.squeeze(-1)
    return points2D  # Return the projected points in 2D (batch_size,4,2)
