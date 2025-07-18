from typing import Sequence

import cv2
import numpy as np
from numpy import ndarray


def solve_reformat_p3p_solutions(
    points_3D: ndarray, points_2D: ndarray, A: ndarray
) -> ndarray:
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
