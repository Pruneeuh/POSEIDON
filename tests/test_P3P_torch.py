import pytest
from poseidon.numpy.utils.initialize_camera_parameters import *
from poseidon.numpy.utils.points import *
from poseidon.torch.utils.before_p3p import convert_matrix_numpy_to_batch, compute_features_vectors
from poseidon.torch.p3p.p3p import P3P
from poseidon.numpy.p3p.p3p import find_best_solution_P3P

precision = 1e-6

@pytest.mark.parametrize("_",range(10))
def test_P3P_estimation_points(_):
    '''
    Test the P3P algorithm with random parameters and check if the points 2D are correctly 
    projected from 3D points with the estimated camera parameters.
    '''
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
    features_vectors = compute_features_vectors(points_3D, C, R)  # Compute features vectors (batch_size, 3, 3)

    # Apply P3P algorithm
    solutions_P3P = P3P(points_3D[:3], features_vectors).squeeze(0)  # Compute P3P solutions (batch_size, 4, 3, 4)
    solutions_P3P_np = solutions_P3P.detach().numpy()  # Convert to numpy array (4*3*4)
    R_opti_torch, C_opti_torch, error = find_best_solution_P3P(points_2D_np, points_3D_np, solutions_P3P_np,A_np)  # Find the best solution from P3P estimation

    assert error < precision, "Error in P3P estimation of the 2D point is too high"

