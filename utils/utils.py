import torch

def mahalanobis(u, v, covarience):
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(covarience), delta))
    return torch.sqrt(m)

def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)
