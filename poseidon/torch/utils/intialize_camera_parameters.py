import torch
from poseidon.numpy.utils.initialize_camera_parameters import (
    generate_camera_parameters,
    generate_position_matrix,
    generate_rotation_matrix,
)
from torch import Tensor


def generate_camera_parameters_batch(batch_size : int) -> Tensor:
    """Generate random camera parameters.
    Args:
        batch_size (int): Number of camera parameters matrix to generate.
    Returns:
        A (torch.Tensor): Intrinsic camera matrix of shape (batch_size,3, 3).
    """
    A: Tensor = torch.empty(batch_size, 3, 3, dtype=torch.float64)

    for i in range(batch_size):
        A[i] = torch.tensor(generate_camera_parameters(), dtype=torch.float64)

    return A


def generate_position_matrix_batch(batch_size : int) -> Tensor:
    """Generate random position vectors.
    Args:
        batch_size (int): Number of position vectors to generate.
    Returns:
        C (torch.Tensor): Position vector of shape (batch_size,3, 1).
    """
    C: Tensor = torch.empty(batch_size, 3, 1, dtype=torch.float64)

    for i in range(batch_size):
        C[i] = torch.tensor(generate_position_matrix(), dtype=torch.float64)

    return C


def generate_rotation_matrix_batch(batch_size : int) -> Tensor:
    """Generate random rotation matrices.
    Args:
        batch_size (int): Number of rotation matrices to generate.
    Returns:
        R (torch.Tensor): Rotation matrix of shape (batch_size,3, 3).
    """
    R: Tensor = torch.empty(batch_size, 3, 3, dtype=torch.float64)

    for i in range(batch_size):
        R[i] = torch.tensor(generate_rotation_matrix(), dtype=torch.float64)

    return R
