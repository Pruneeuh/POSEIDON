import time

from poseidon.numpy.p3p import solve_reformat_p3p_solutions
from poseidon.torch.p3p import P3P
from poseidon.torch.utils import *
from poseidon.torch.utils import (
    compute_features_vectors,
    generate_points_3D,
    projection_all_point3D_to2D,
)

precision = 1e-6
nb_tests = 10
batch_size = 5000


def test_P3P_estimation_points():
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
    features_vectors = compute_features_vectors(points_3D[:, :3], C, R)  # (batch_size, 3, 3)

    # Apply P3P algorithm
    start_torch = time.time()
    solutions_P3P_torch = P3P(points_3D[:, :3], features_vectors)
    end_torch = time.time()

    total_time_torch = end_torch - start_torch
    total_time_opencv = 0

    for i in range(batch_size):
        A_actual = A[i].detach().numpy()
        points_2D_actual = points_2D[i].detach().numpy()
        points_3D_actual = points_3D[i].detach().numpy()
        start_opencv = time.time()
        solutions_P3P_opencv = solve_reformat_p3p_solutions(
            points_3D_actual, points_2D_actual, A_actual
        )
        end_opencv = time.time()
        total_time_opencv += end_opencv - start_opencv

    print(f"Total time for P3P in PyTorch: {total_time_torch:.4f} seconds")
    print(f"Total time for P3P in OpenCV: {total_time_opencv:.4f} seconds")


test_P3P_estimation_points()
