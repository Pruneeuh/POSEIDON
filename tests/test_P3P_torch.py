import pytest
from poseidon.numpy.p3p.p3p import find_best_solution_P3P
from poseidon.numpy.utils.before_p3p import *
from poseidon.numpy.utils.initialize_camera_parameters import *
from poseidon.torch.p3p.p3p import P3P, solve_reformat_p3p_solutions
from poseidon.torch.utils.before_p3p import (
    compute_features_vectors,
    convert_matrix_numpy_to_batch,
)

precision = 1e-6
nb_tests = 10


@pytest.mark.parametrize("_", range(nb_tests))
def test_P3P_estimation_points(_):
    """
    Test the P3P algorithm in torch using random parameters by evaluating the projection
    error between the original 2D image points and the 2D points projected from
    3D points using the estimated camera parameters.
    """
    # Generate random camera parameters
    C_np = generate_position_matrix()
    R_np = generate_rotation_matrix()
    A_np = generate_camera_parameters()

    # Generate random points
    points_3D_np = generate_points_3D()
    points_2D_np = projection_points_2D(points_3D_np, C_np, R_np, A_np)

    # Convert numpy arrays to torch tensors with batch size 1
    C = convert_matrix_numpy_to_batch(C_np)
    R = convert_matrix_numpy_to_batch(R_np)
    points_3D = convert_matrix_numpy_to_batch(points_3D_np)
    print("points_3D = \n", points_3D)

    # Compute features vectors
    features_vectors = compute_features_vectors(points_3D, C, R)  #  (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors).squeeze(0)  # (batch_size, 4, 3, 4)
    solutions_P3P_np = solutions_P3P.detach().numpy()  # Convert to numpy array (4*3*4)

    # Find the best solution from P3P estimation
    R_opti_torch, C_opti_torch, error = find_best_solution_P3P(
        points_2D_np, points_3D_np, solutions_P3P_np, A_np
    )

    assert error < precision, "Error in P3P estimation of the 2D point is too high"


@pytest.mark.parametrize("_", range(nb_tests))
def test_P3P_rotation_position(_):
    """
    Test the P3P algorithm in torch using random parameters by evaluating the error
    between the ground truth and the estimated camera pose (position and rotation)
    obtained from the P3P solution.
    """
    # Generate random camera parameters
    C_np = generate_position_matrix()
    R_np = generate_rotation_matrix()
    A_np = generate_camera_parameters()

    # Generate random points
    points_3D_np = generate_points_3D()
    points_2D_np = projection_points_2D(points_3D_np, C_np, R_np, A_np)

    # Convert numpy arrays to torch tensors with batch size 1
    C = convert_matrix_numpy_to_batch(C_np)
    R = convert_matrix_numpy_to_batch(R_np)
    points_3D = convert_matrix_numpy_to_batch(points_3D_np)
    print("points_3D = \n", points_3D)

    # Compute features vectors
    features_vectors = compute_features_vectors(points_3D, C, R)  # (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors).squeeze(0)  # (batch_size, 4, 3, 4)
    solutions_P3P_np = solutions_P3P.detach().numpy()  # Convert to numpy array (4*3*4)

    # Find the best solution from P3P estimation
    R_opti_torch, C_opti_torch, error = find_best_solution_P3P(
        points_2D_np, points_3D_np, solutions_P3P_np, A_np
    )
    assert np.allclose(
        R_opti_torch, R_np, atol=precision
    ), "Estimated rotation does not match the original rotation"
    assert np.allclose(
        C_opti_torch, C_np, atol=precision
    ), "Estimated position does not match the original position"


@pytest.mark.parametrize("_", range(nb_tests))
def test_P3P_opencv(_):
    """
    Test the P3P algorithm using random parameters by comparing
    the camera pose (position and rotation) estimated by OpenCV’s
    P3P implementation and the one estimated by the Poseidon implementation,
    based on the error between both results.
    """
    # Generate random camera parameters
    C_np = generate_position_matrix()
    R_np = generate_rotation_matrix()
    A_np = generate_camera_parameters()

    # Generate random points
    points_3D_np = generate_points_3D()
    points_2D_np = projection_points_2D(points_3D_np, C_np, R_np, A_np)

    # Convert numpy arrays to torch tensors with batch size 1
    C = convert_matrix_numpy_to_batch(C_np)
    R = convert_matrix_numpy_to_batch(R_np)
    points_3D = convert_matrix_numpy_to_batch(points_3D_np)
    print("points_3D = \n", points_3D)

    # Compute features vectors
    features_vectors = compute_features_vectors(points_3D, C, R)  # (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors).squeeze(0)  # (batch_size, 4, 3, 4)
    solutions_P3P_np = solutions_P3P.detach().numpy()  # Convert to numpy array (4*3*4)

    # Find the best solution from P3P estimation
    R_opti_torch, C_opti_torch, error = find_best_solution_P3P(
        points_2D_np, points_3D_np, solutions_P3P_np, A_np
    )

    # Apply OpenCV's P3P
    solutions_opencv = solve_reformat_p3p_solutions(points_3D_np[:3], points_2D_np[:3], A_np)
    R_opti_opencv, C_opti_opencv, erreur = find_best_solution_P3P(
        points_2D_np, points_3D_np, solutions_opencv, A_np
    )

    assert np.allclose(
        R_opti_torch, R_opti_opencv, atol=precision
    ), "Estimated rotation by Poseidon does not match OpenCV's P3P implementation"
    assert np.allclose(
        C_opti_torch, C_opti_opencv, atol=precision
    ), "Estimated position by Poseidon does not match OpenCV's P3P implementation"
