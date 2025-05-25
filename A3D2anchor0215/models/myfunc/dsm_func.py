import torch
import torch.nn as nn


def  dsm(matrix, max_iter=10, epsilon=1e-6):
    for _ in range(max_iter):
        # Normalize rows
        matrix = matrix / matrix.sum(dim=1, keepdim=True)
        # Normalize columns
        matrix = matrix / matrix.sum(dim=0, keepdim=True)
        # Check for convergence (optional, for efficiency)
        if torch.allclose(matrix.sum(dim=1), torch.ones_like(matrix.sum(dim=1)), atol=epsilon) and \
            torch.allclose(matrix.sum(dim=0), torch.ones_like(matrix.sum(dim=0)), atol=epsilon):
            break

    return matrix