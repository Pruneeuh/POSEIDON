import pytest
from poseidon.numpy import (
    P3P,
    compute_features_vectors,
    find_best_solution_P3P,
    generate_camera_parameters,
    generate_points_3D,
    generate_position_matrix,
    generate_rotation_matrix,
    projection_points_2D,
    solve_reformat_p3p_solutions,
)

precision = 1e-6
nb_tests = 10


@pytest.mark.parametrize("_", range(nb_tests))
def test_P3P_estimation_points(_):
    """
    Test the P3P algorithm in numpy using random parameters by evaluating the projection
    error between the original 2D image points and the 2D points projected from
    3D points using the estimated camera parameters.
    """
    # Generate random camera parameters
    C = generate_position_matrix()
    R = generate_rotation_matrix()
    A = generate_camera_parameters()

    # Generate random points
    points_3D = generate_points_3D()
    points_2D = projection_points_2D(points_3D, C, R, A)

    # Compute features vectors
    features_vectors = compute_features_vectors(points_3D, C, R)  # (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors)

    # Find the best solution from P3P estimation
    R_opti, C_opti, error = find_best_solution_P3P(points_2D, points_3D, solutions_P3P, A)
    print("error = ", error)

    assert error < precision, "Error in P3P estimation of the 2D point is too high"


@pytest.mark.parametrize("_", range(nb_tests))
def test_P3P_rotation_position(_):
    """
    Test the P3P algorithm in numpy using random parameters by evaluating the error
    between the ground truth and the estimated camera pose (position and rotation)
    obtained from the P3P solution.
    """
    # Generate random camera parameters
    C = generate_position_matrix()
    R = generate_rotation_matrix()
    A = generate_camera_parameters()

    # Generate random points
    points_3D = generate_points_3D()
    points_2D = projection_points_2D(points_3D, C, R, A)

    # Compute features vectors
    features_vectors = compute_features_vectors(points_3D, C, R)  # (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors)
    print("solutions_P3P = \n", solutions_P3P)

    # Find the best solution from P3P estimation
    R_opti, C_opti, error = find_best_solution_P3P(points_2D, points_3D, solutions_P3P, A)
    print("R_opti = \n", R_opti)
    print("R = \n", R)

    assert np.allclose(
        R_opti, R, atol=precision
    ), "Estimated rotation does not match the original rotation"
    assert np.allclose(
        C_opti, C, atol=precision
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
    C = generate_position_matrix()
    R = generate_rotation_matrix()
    A = generate_camera_parameters()

    # Generate random points
    points_3D = generate_points_3D()
    points_2D = projection_points_2D(points_3D, C, R, A)

    features_vectors = compute_features_vectors(
        points_3D, C, R
    )  # Compute features vectors (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors)
    print("solutions_P3P = \n", solutions_P3P)

    # Find the best solution from P3P estimation
    R_opti_np, C_opti_np, error = find_best_solution_P3P(points_2D, points_3D, solutions_P3P, A)

    # Apply OpenCV's P3P
    solutions_opencv = solve_reformat_p3p_solutions(points_3D[:3], points_2D[:3], A)
    R_opti_opencv, C_opti_opencv, erreur = find_best_solution_P3P(
        points_2D, points_3D, solutions_opencv, A
    )
    print("R= \n", R)
    print("R_opti_np = \n", R_opti_np)
    print("R_opti_opencv = \n", R_opti_opencv)

    assert np.allclose(
        R_opti_np, R_opti_opencv, atol=1e-3
    ), "Estimated rotation by Poseidon does not match OpenCV's P3P implementation"
    assert np.allclose(
        C_opti_np, C_opti_opencv, atol=1e-3
    ), "Estimated position by Poseidon does not match OpenCV's P3P implementation"
