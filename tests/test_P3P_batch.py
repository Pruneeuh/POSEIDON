import pytest
from poseidon.torch.p3p.p3p import P3P, find_best_solution_P3P_batch
from poseidon.torch.utils.before_p3p import (
    compute_features_vectors,
    generate_points_3D,
    projection_all_point3D_to2D,
)
from poseidon.torch.utils.intialize_camera_parameters import *

precision = 1e-6
nb_tests = 10
batch_size = 5


@pytest.mark.parametrize("_", range(nb_tests))
def test_P3P_estimation_points(_):
    """
    Test the P3P algorithm in PyTorch with batched inputs using random parameters.

    The evaluation is performed by computing the projection error between the original
    2D image points and the 2D points reprojected from the 3D points using the estimated
    camera parameters.
    """
    # Generate random camera parameters
    C = generate_position_matrix_batch(batch_size)
    R = generate_rotation_matrix_batch(batch_size)
    A = generate_camera_parameters_batch(batch_size)

    # Generate random points
    points_3D = generate_points_3D(batch_size)
    points_2D = projection_all_point3D_to2D(points_3D, C, R, A)

    # Compute features vectors
    features_vectors = compute_features_vectors(points_3D, C, R)  # (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors)

    # Find the best solution from P3P estimation
    R_opti, C_opti, error = find_best_solution_P3P_batch(solutions_P3P, points_3D, points_2D)
