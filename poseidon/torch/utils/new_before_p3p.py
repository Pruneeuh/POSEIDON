import torch
from torch import Tensor

def get_feature_vectors(points2D : Tensor, A : Tensor) -> Tensor:
    """
    Compute feature vectors from 2D points and intrinsic matrix.
    
    Args:
        points2D (torch.Tensor): 2D points in image coordinates (batch_size,3,2).
        A (torch.Tensor): Camera intrinsic matrix (batch_size,3,3).
        
    Returns:
        featuresVect (torch.tensor):  feature vectors for each point (batch_size, 3, 3).)
    """
    batch_size :int = points2D.shape[0]  
    ones : Tensor = torch.ones((batch_size, 3,1), dtype=torch.float64) #(batch_size, 3, 1)

    # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
    p_h : Tensor = torch.cat([points2D, ones], dim=-1) # (batch_size, 3, 3)

    A_inv : Tensor = torch.linalg.inv(A)  # Inverse of intrinsic matrix

    featuresVectList : list = []

    for i in range(3):
        p_h_i : Tensor= p_h[:,i].unsqueeze(-1) # (batch_size, 3,1)

        # Apply inverse of intrinsic matrix to get direction vector in camera frame
        fi : Tensor = torch.matmul(A_inv, p_h_i).squeeze(-1)  # (batch_size, 3)

        # Normalize to get a unit vector (bearing direction)
        fi = fi / torch.norm(fi, dim=1,keepdim=True)  # Normalize along the first dimension

        featuresVectList.append(fi)  # Reshape to (batch_size, 3, 1)
    
    # Stack into a matrix: shape (3, 3) where each column is f1, f2, f3
    featuresVect : Tensor = (torch.stack(featuresVectList, dim=1))  # (batch_size, 3, 3)
    return featuresVect