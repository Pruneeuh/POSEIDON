import torch
from torch import Tensor


def P3P(points_3D: Tensor, features_vectors: Tensor) -> Tensor:
    """
    This function computes the P3P alorithm.
    Args:
        points_3D : tensor with the 3D points : (batch_size,3,3)
        features_vectors : tensor with the features vectors : (batch_size,3,3)
    Returns:
        solutions : tensor with the solutions of the P3P algorithm : (batch_size,4,3,4)
    """
    batch_size: int = points_3D.shape[0]  # Get the batch size from the first dimension of points_3D

    # Extract the 3D points
    P1: Tensor = points_3D[:, 0, :]  # (batch_size,3)
    P2: Tensor = points_3D[:, 1, :]  # (batch_size,3)
    P3: Tensor = points_3D[:, 2, :]  # (batch_size,3)

    # Extract the features vectors
    f1: Tensor = features_vectors[:, 0, :]  # (batch_size,3)
    f2: Tensor = features_vectors[:, 1, :]  # (batch_size,3)
    f3: Tensor = features_vectors[:, 2, :]  # (batch_size,3)

    # Creation of the solutions tensor
    solutions: Tensor = torch.empty((batch_size, 4, 3, 4), dtype=torch.float64)

    # Verification that the points are not collinear
    v1: Tensor = P2 - P1  # (batch_size, 3)
    v2: Tensor = P3 - P1  # (batch_size, 3)
    norms = torch.norm(torch.cross(v1, v2, dim=1), dim=1)
    all_dif_zero = torch.all(norms != 0)

    if not all_dif_zero:
        raise ValueError("The points must not be collinear")

    # Creation of an orthonormal frame (from f1, f2 and f3)
    # The frame T = (C,tx,ty,tz)
    tx: Tensor = f1  # (batch_size,3)
    tz: Tensor = torch.cross(f1, f2, dim=1) / torch.norm(
        torch.cross(f1, f2, dim=1), dim=1, keepdim=True
    )
    # (batch_size,3)
    ty: Tensor = torch.cross(tz, tx, dim=1)  # (batch_size,3)

    # Reshaping the vectors to (batch_size,1,3) for matrix operations
    tx = torch.reshape(tx, (batch_size, 1, 3))  # (batch_size,1,3)
    ty = torch.reshape(ty, (batch_size, 1, 3))  # (batch_size,1,3)
    tz = torch.reshape(tz, (batch_size, 1, 3))  # (batch_size,1,3)

    # Creation of a transformation matrix T and expression of the f3 vector in this frame
    T: Tensor = torch.cat((tx, ty, tz), dim=1)  # (3*3)
    f3_T: Tensor = torch.matmul(T, f3.unsqueeze(-1))  # (batch_size,3,1)

    # Check if the f3 vector is positive in the T frame (for sign of teta later)
    f3_T_positif: Tensor = f3_T[:, 2] > 0  # (batch_size,1)

    # Calculation of vectors of the base η = (P1,nx,ny,nz)
    nx: Tensor = (P2 - P1) / torch.norm(P2 - P1, dim=1, keepdim=True)  # (batch_size,3)
    nz: Tensor = torch.cross(nx, P3 - P1, dim=1) / torch.norm(
        torch.cross(nx, P3 - P1, dim=1), dim=1, keepdim=True
    )  # (batch_size,3)
    ny: Tensor = torch.cross(nz, nx, dim=1)  # (batch_size,3)

    # Reshape the vectors to (1,3) for concatenation
    nx = torch.reshape(nx, (batch_size, 1, 3))  # (batch_size,1,3)
    ny = torch.reshape(ny, (batch_size, 1, 3))
    nz = torch.reshape(nz, (batch_size, 1, 3))

    # Computation of the matrix N and the world point P3
    N: Tensor = torch.cat((nx, ny, nz), dim=1)  #  T's equivalent in the world coordinate system
    P3_N: Tensor = torch.matmul(N, (P3 - P1).unsqueeze(-1))

    # Computation of phi1 et phi2 with 0=x, 1=y, 2=z
    phi1: Tensor = f3_T[:, 0] / f3_T[:, 2]  # (batch_size,1)
    phi2: Tensor = f3_T[:, 1] / f3_T[:, 2]  # (batch_size,1)

    # Extraction of p1 and p2 from P3_eta
    p1: Tensor = P3_N[:, 0]  # x  # (batch_size,3)
    p2: Tensor = P3_N[:, 1]  # y  # (batch_size,3)

    # Computation of d12
    d12: Tensor = torch.norm(P2 - P1, dim=1, keepdim=True)  # (batch_size,1)

    # Computation of b = cot(beta)
    cosBeta: Tensor = (
        torch.sum(f1 * f2, dim=1) / (torch.norm(f1, dim=1) * torch.norm(f2, dim=1))
    ).unsqueeze(
        -1
    )  # tensor.dot(a,b) <=> tensor.sum(a*b)   # (batch_size,1)

    b: Tensor = torch.sqrt(1 / (1 - cosBeta**2) - 1)
    b = torch.where(cosBeta < 0, -b, b)  # If cosBeta < 0, then b = -b    # (batch_size,1)

    # Calculation of the coefficients of the polynomial
    a4: Tensor = -(phi2**2) * p2**4 - phi1**2 * p2**4 - p2**4
    a3: Tensor = (
        2 * p2**3 * d12 * b + 2 * phi2**2 * p2**3 * d12 * b - 2 * phi1 * phi2 * p2**3 * d12
    )
    a2: Tensor = (
        -(phi2**2) * p1**2 * p2**2
        - phi2**2 * p2**2 * d12**2 * b**2
        - phi2**2 * p2**2 * d12**2
        + phi2**2 * p2**4
        + phi1**2 * p2**4
        + 2 * p1 * p2**2 * d12
        + 2 * phi1 * phi2 * p1 * p2**2 * d12 * b
        - phi1**2 * p1**2 * p2**2
        + 2 * phi2**2 * p1 * p2**2 * d12
        - p2**2 * d12**2 * b**2
        - 2 * p1**2 * p2**2
    )
    a1: Tensor = (
        2 * p1**2 * p2 * d12 * b
        + 2 * phi1 * phi2 * p2**3 * d12
        - 2 * phi2**2 * p2**3 * d12 * b
        - 2 * p1 * p2 * d12**2 * b
    )
    a0: Tensor = (
        -2 * phi1 * phi2 * p1 * p2**2 * d12 * b
        + phi2**2 * p2**2 * d12**2
        + 2 * p1**3 * d12
        - p1**2 * d12**2
        + phi2**2 * p1**2 * p2**2
        - p1**4
        - 2 * phi2**2 * p1 * p2**2 * d12
        + phi1**2 * p1**2 * p2**2
        + phi2**2 * p2**2 * d12**2 * b**2
    )

    # Computation of the roots of the polynomial
    roots: Tensor = polynomial_root_calculation_4th_degree_ferrari(
        a0, a1, a2, a3, a4
    )  # (batch_size,4,2)

    # For each solution : comptation of the camera position and rotation matrix
    for i in range(4):
        # Computation of trigonometrics forms
        cos_teta: Tensor = (roots[:, i, 0]).unsqueeze(-1)  # real part of the root (batch_size,1)
        sin_teta: Tensor = torch.where(
            f3_T_positif, -torch.sqrt(1 - cos_teta**2), torch.sqrt(1 - cos_teta**2)
        )

        cot_alpha: Tensor = ((phi1 / phi2) * p1 + cos_teta * p2 - d12 * b) / (
            (phi1 / phi2) * cos_teta * p2 - p1 + d12
        )
        sin_alpha: Tensor = torch.sqrt(1 / (cot_alpha**2 + 1))
        cos_alpha: Tensor = torch.sqrt(1 - sin_alpha**2)
        cos_alpha = torch.where(cot_alpha < 0, -cos_alpha, cos_alpha)

        # Computation of the intermediate rotation's matrixs
        C_estimate: Tensor = torch.stack(
            [
                d12 * cos_alpha * (sin_alpha * b + cos_alpha),
                d12 * sin_alpha * cos_teta * (sin_alpha * b + cos_alpha),
                d12 * sin_alpha * sin_teta * (sin_alpha * b + cos_alpha),
            ],
            dim=1,
        )  # (batch_size,3,1)
        # (batch_size,3,1)

        Q_row1: Tensor = torch.stack(
            [-cos_alpha, -sin_alpha * cos_teta, -sin_alpha * sin_teta], dim=-1
        )
        Q_row2: Tensor = torch.stack(
            [sin_alpha, -cos_alpha * cos_teta, -cos_alpha * sin_teta], dim=-1
        )
        Q_row3: Tensor = torch.stack([0 * sin_teta, -sin_teta, cos_teta], dim=-1)
        Q: Tensor = torch.stack([Q_row1, Q_row2, Q_row3], dim=1).squeeze(2)  # (batch_size,3*3)

        # Computation of the absolute camera center
        C_estimate = P1.unsqueeze(-1) + torch.matmul(
            torch.transpose(N, 1, 2), C_estimate
        )  # (batch_size,3,1)

        # Computation of the orientation matrix
        R_estimate: Tensor = torch.matmul(
            torch.matmul(torch.transpose(N, 1, 2), torch.transpose(Q, 1, 2)), T
        )  # (batch_size,3,3)
        R_estimate = torch.transpose(R_estimate, 1, 2)  # (batch_size,3,3) ??????

        # Adding C and R to the solutions
        solutions[:, i, :, :1] = C_estimate
        solutions[:, i, :, 1:] = R_estimate

    return solutions  # Return the solutions of the P3P algorithm (batch_size,4,3,4)


def find_best_solution_P3P_batch(
    solutions: Tensor, points2D: Tensor, points3D: Tensor, A: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    This function finds the best solution from the P3P algorithm.
    Args:
        solutions (torch.Tensor): Solution matrix returned by P3P (batch_size, 4, 3, 4).
        points2D (torch.Tensor): 2D points used for P3P of shape (batch_size,4,2).
        points3D (torch.Tensor): 3D points used for P3P of shape (batch_size,4,3).
        A (torch.Tensor): Camera parameters matrix (batch_size, 3, 3).
    Returns:
        R_opti (torch.Tensor): Optimal rotation matrix of shape (batch_size,3, 3).
        C_opti (torch.Tensor): Optimal position vector of shape (batch_size,3, 1).
        error (torch.Tensor): Error of the best solution (batch_size,1).
    """
    batch_size: int = solutions.shape[0]  # Get the batch size from the first dimension of solutions

    erreurs: Tensor = torch.zeros(batch_size, 4)  # (batch_size,4)

    for i in range(4):  # Iterate over the 4 solutions
        R: Tensor = solutions[:, i, :, 1:]  # (batch_size,3,3)
        C: Tensor = solutions[:, i, :, :1]  # (batch_size,3,1)

        is_nan: Tensor = torch.isnan(R).any(
            dim=(1, 2)
        )  # Check if any element in R is NaN (batch_size,)

        points_2D_P3P: Tensor = projection_all_point3D_to2D(points3D, C, R, A)  # (batch_size,4,2)

        distance: Tensor = torch.norm(points2D - points_2D_P3P, dim=2)  # (batch_size,4)
        erreur_totale: Tensor = torch.sum(
            distance, dim=1
        )  # Sum the distances for each solution (batch_size,)
        erreur_totale = torch.where(is_nan, float("inf"), erreur_totale)
        erreurs[:, i] = erreur_totale  # Store the error for the current solution (batch_size,)

    # Find the best solution (with the smallest estimation error)
    value_min: Tensor
    index_min: Tensor
    value_min, index_min = torch.min(erreurs, dim=1, keepdim=True)  # (batch_size,1)

    best_solutions: Tensor = solutions[
        torch.arange(batch_size), index_min.squeeze(-1)
    ]  # (batch_size,3,4)

    R_opti: Tensor = best_solutions[:, :, 1:]  # Optimal rotation matrix (batch_size,3,3)
    C_opti: Tensor = best_solutions[:, :, :1]  # Optimal position vector (batch_size,3,1)

    return (
        R_opti,
        C_opti,
        value_min,
    )  # Return the optimal rotation matrix, position vector and the error of the best solution
