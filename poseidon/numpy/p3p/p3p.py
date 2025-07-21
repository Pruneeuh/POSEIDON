from typing import Sequence

import cv2
import numpy as np
from numpy import ndarray
from poseidon.numpy.utils.points import projection_points_2D


def solve_reformat_p3p_solutions(points_3D: ndarray, points_2D: ndarray, A: ndarray) -> ndarray:
    """Reformat the P3P solutions from opencv to a specific structure.
    Args:
        points_3D (np.ndarray): 3 3D points of shape (3, 3).
        points_2D (np.ndarray): Projected 2D points of shape (3, 2).
        A (np.ndarray): Intrinsic camera matrix of shape (3, 3).
    Returns:
        solutions (np.ndarray): Reformatted solutions of shape (4, 3, 4).
    """
    nbsol: int
    rvec: Sequence[ndarray]
    tvecs: Sequence[ndarray]
    nbsol, rvec, tvecs = cv2.solveP3P(
        points_3D,
        points_2D,
        A,
        np.zeros(
            4,
        ),
        flags=cv2.SOLVEPNP_P3P,
    )  # Solve P3P problem using OpenCV

    solutions: ndarray = np.zeros((4, 3, 4))  # (4*3*4)

    for i in range(len(rvec)):
        # Transition from the Rodriguez vectors to the rotation matrix
        rodriguez: ndarray = rvec[i]  # (3*1)
        R_P3P: ndarray = cv2.Rodrigues(rodriguez)[0]  # rotation matrix : (3*3)

        C_P3P: ndarray = tvecs[i]  # translation matrix : (3*1)

        solutions[i, :, :1] = C_P3P
        solutions[i, :, 1:] = R_P3P
    return solutions


def P3P(points_3D: ndarray, features_vectors: ndarray) -> ndarray:
    """Solve the P3P problem using OpenCV.
    Args:
        points_3D (np.ndarray): 4 3D points of shape (4, 3).
        features_vectors (np.ndarray): array of features vectors of shape (3, 3).
    Returns:
        solutions (np.ndarray): Solutions of shape (4, 3, 4).
    """

    # Extract the 3D points
    P1 = points_3D[0]
    P2 = points_3D[1]
    P3 = points_3D[2]

    # Extract the features vectors
    f1 = features_vectors[0]
    f2 = features_vectors[1]
    f3 = features_vectors[2]

    # Creation of the solutions tensor
    solutions = np.zeros((4, 3, 4))

    # Verification that the points are not collinear
    v1 = P2 - P1
    v2 = P3 - P1
    if np.linalg.norm(np.cross(v1, v2)) == 0:
        raise ValueError("The points must not be collinear")

    # Creation of an orthonormal frame (from f1, f2 and f3)
    # The frame T = (C,tx,ty,tz)
    tx = f1
    tz = np.cross(f1, f2) / np.linalg.norm(np.cross(f1, f2))
    ty = np.cross(tz, tx)

    # Reshaping the vectors to (batch_size,1,3) for matrix operations
    tx = np.reshape(tx, (1, 3))  # (1*3)
    ty = np.reshape(ty, (1, 3))
    tz = np.reshape(tz, (1, 3))

    # Creation of a transformation matrix T and expression of the f3 vector in this frame
    T = np.concatenate((tx, ty, tz), axis=0)  # (3*3)
    f3_T = np.dot(T, f3)  # (3,)

    # Check if the f3 vector is positive in the T frame (for sign of teta later)
    f3_T_positif = False
    if f3_T[2] > 0:
        f3_T_positif = True

    # Calculation of vectors of the base η = (P1,nx,ny,nz)
    nx = (P2 - P1) / np.linalg.norm(P2 - P1)  # (3,)
    nz = np.cross(nx, P3 - P1) / np.linalg.norm(np.cross(nx, P3 - P1))
    ny = np.cross(nz, nx)

    # Reshape the vectors to (1,3) for concatenation
    nx = np.reshape(nx, (1, 3))  # (1,3)
    ny = np.reshape(ny, (1, 3))
    nz = np.reshape(nz, (1, 3))

    # Computation of the matrix N and the world point P3
    N = np.concatenate((nx, ny, nz), axis=0)  # (3*3) T's equivalent in the world coordinate system
    P3_N = np.dot(N, P3 - P1)  # (3,)

    # Computation of phi1 et phi2 with f3_T = [f3,x ; f3,y ; f3,z]
    phi1 = f3_T[0] / f3_T[2]
    phi2 = f3_T[1] / f3_T[2]

    # Extraction of p1 and p2 from P3_eta
    p1 = P3_N[0]  # x
    p2 = P3_N[1]  # y

    # Computation of d12
    d12 = np.linalg.norm(P2 - P1)

    # Computation of b = cot(beta)
    cosBeta = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

    b = np.sqrt(1 / (1 - cosBeta**2) - 1)

    if cosBeta < 0:
        b = -b
    print("b = ", b)
    print(type(b))

    # Calculation of the coefficients of the polynomial
    a4 = -(phi2**2) * p2**4 - phi1**2 * p2**4 - p2**4
    a3 = 2 * p2**3 * d12 * b + 2 * phi2**2 * p2**3 * d12 * b - 2 * phi1 * phi2 * p2**3 * d12
    a2 = (
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
    a1 = (
        2 * p1**2 * p2 * d12 * b
        + 2 * phi1 * phi2 * p2**3 * d12
        - 2 * phi2**2 * p2**3 * d12 * b
        - 2 * p1 * p2 * d12**2 * b
    )
    a0 = (
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

    roots = polynomial_root_calculation_4th_degree_ferrari(np.array([a0, a1, a2, a3, a4]))

    print("roots = \n", roots)

    # For each solution of the polynomial
    for i in range(4):
        # Computation of trigonometrics forms
        cos_teta = np.real(roots[i])

        if f3_T_positif == True:  # teta in [-pi,0]
            sin_teta = -np.sqrt(1 - cos_teta**2)
        else:  # f3_T négatif donc teta in [0,pi]
            sin_teta = np.sqrt(1 - cos_teta**2)

        cot_alpha = ((phi1 / phi2) * p1 + cos_teta * p2 - d12 * b) / (
            (phi1 / phi2) * cos_teta * p2 - p1 + d12
        )
        print("cot_alpha = ", cot_alpha)

        sin_alpha = np.sqrt(1 / (cot_alpha**2 + 1))
        cos_alpha = np.sqrt(1 - sin_alpha**2)

        if cot_alpha < 0:
            cos_alpha = -cos_alpha

        # Computation of the intermediate rotation's matrixs
        C_estimate = [
            d12 * cos_alpha * (sin_alpha * b + cos_alpha),
            d12 * sin_alpha * cos_teta * (sin_alpha * b + cos_alpha),
            d12 * sin_alpha * sin_teta * (sin_alpha * b + cos_alpha),
        ]  # (3,)

        Q = [
            [-cos_alpha, -sin_alpha * cos_teta, -sin_alpha * sin_teta],
            [sin_alpha, -cos_alpha * cos_teta, -cos_alpha * sin_teta],
            [0, -sin_teta, cos_teta],
        ]  # (3*3)

        # Computation of the absolute camera center
        C_estimate = P1 + np.transpose(N) @ C_estimate  # (3,)
        C_estimate = C_estimate[:, np.newaxis]  # (3,1)

        # Computation of the orientation matrix
        R_estimate = np.transpose(N) @ np.transpose(Q) @ T  # (3*3)

        # Adding C and R to the solutions
        solutions[i, :, :1] = C_estimate
        solutions[i, :, 1:] = np.transpose(R_estimate)

        return solutions


def print_best_solution_P3P(
    points_2D: ndarray, points_3D: ndarray, solutions: ndarray, A: ndarray
) -> tuple[ndarray, ndarray]:
    """Find the best solution from P3P estimation based on the smallest error.
    Args:
        points_2D (np.ndarray): Projected 2D points of shape (4, 2).
        points_3D (np.ndarray): 3D points of shape (4, 3).
        solutions (np.ndarray): Solutions from P3P of shape (4, 3, 4).
    Returns:
        R_opti (np.ndarray): Optimal rotation matrix of shape (3, 3).
        C_opti (np.ndarray): Optimal position vector of shape (3, 1).
    """
    erreurs: list = []
    nb_sol: int = 0

    for i in range(len(solutions)):
        R: ndarray = solutions[i, :, 1:]  # Rotation matrix (3*3)
        C: ndarray = solutions[i, :, :1]  # Position matrix (3*1)

        # if not np.all(R==np.zeros((3,3))) and not np.any(np.isnan(R)) :  # Check if R is not a zero matrix and does not contain NaN values
        nb_sol += 1
        print("------------ Solution n° : ", nb_sol, "----------------")
        erreurs.append([0.0])
        if not np.isnan(R[0, 0]) and not np.all(
            R == np.zeros((3, 3))
        ):  # Check if R is not a zero matrix and does not contain NaN values
            print(
                "R = \n",
                R,
            )
            print(
                "C = \n",
                C,
            )
            points_2D_P3P: ndarray = projection_points_2D(
                points_3D, C, R, A
            )  # Project the 3D points to 2D using the P3P solution
            for j in range(len(points_2D)):
                erreur_pt: float = np.linalg.norm(
                    points_2D_P3P[j, :] - points_2D[j, :]
                )  # conversion in float for mypy
                print("erreur P", j + 1, " = ", type(erreur_pt))
                erreurs[i] += erreur_pt

        else:
            print("matrix R is a zero matrix or contains NaN values, skipping this solution.")
            erreurs[i] = float("inf")  # Handle NaN values
            print("erreur P3P = inf")

    # Find the best solution (with the smallest estimation error)
    indice_min: int = 0
    min: float = erreurs[0]
    for i in range(1, len(erreurs)):
        if erreurs[i] < min:
            min = erreurs[i]
            indice_min = i

    R_opti: ndarray = solutions[indice_min, :, 1:]
    C_opti: ndarray = solutions[indice_min, :, :1]

    print("\n------------ Best solution : ----------------")
    print("Solution n° :", indice_min + 1, "\n")
    print("R estimé = \n", R_opti, "\n")
    print("C estimé = \n", C_opti, "\n")
    print("Erreur totale = ", min, "\n")

    return R_opti, C_opti


def find_best_solution_P3P(
    points_2D: ndarray, points_3D: ndarray, solutions: ndarray, A: ndarray
) -> tuple[ndarray, ndarray, float]:
    """Find the best solution from P3P estimation based on the smallest error.
    Args:
        points_2D (np.ndarray): Projected 2D points of shape (4, 2).
        points_3D (np.ndarray): 3D points of shape (4, 3).
        solutions (np.ndarray): Solutions from P3P of shape (4, 3, 4).
    Returns:
        R_opti (np.ndarray): Optimal rotation matrix of shape (3, 3).
        C_opti (np.ndarray): Optimal position vector of shape (3, 1).
        error (float): Error of the best solution.
    """
    erreurs: list = []
    nb_sol: int = 0

    for i in range(len(solutions)):
        R: ndarray = solutions[i, :, 1:]  # Rotation matrix (3*3)
        C: ndarray = solutions[i, :, :1]  # Position matrix (3*1)

        # if not np.all(R==np.zeros((3,3))) and not np.any(np.isnan(R)) :  # Check if R is not a zero matrix and does not contain NaN values
        nb_sol += 1

        erreurs.append([0.0])
        if not np.isnan(R[0, 0]) and not np.all(
            R == np.zeros((3, 3))
        ):  # Check if R is not a zero matrix and does not contain NaN values
            points_2D_P3P: ndarray = projection_points_2D(
                points_3D, C, R, A
            )  # Project the 3D points to 2D using the P3P solution
            for j in range(len(points_2D)):
                erreur_pt: float = np.linalg.norm(points_2D_P3P[j, :] - points_2D[j, :])
                erreurs[i] += erreur_pt

        else:
            erreurs[i] = float("inf")  # Handle NaN values

    # Find the best solution (with the smallest estimation error)
    indice_min: int = 0
    min: float = erreurs[0]
    for i in range(1, len(erreurs)):
        if erreurs[i] < min:
            min = erreurs[i]
            indice_min = i

    R_opti: ndarray = np.transpose(solutions[indice_min, :, 1:])
    C_opti: ndarray = solutions[indice_min, :, :1]

    return (
        R_opti,
        C_opti,
        min,
    )  # Return the optimal rotation matrix, position vector and the error of the best solution
