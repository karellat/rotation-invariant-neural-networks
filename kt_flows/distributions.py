#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch



import torch



def sample_gmm(N, weights, means, stds):
    
    
    weights = weights/weights.sum()
    z = torch.multinomial(weights, N, replacement = True)
    dimension = means.shape[1]
    eps = torch.randn(N,dimension,device=means.device)
    x = means[z] +stds[z].view(N,1)*eps
    return x


import torch


import torch


def sample_gmm_base(N, means, stds):
    """
    means: shape (M, d)
    stds:  shape (M, d)
    """
    M, d = means.shape

    z = torch.randint(M, size=(N,), device=means.device)

    eps = torch.randn(N, d, device=means.device)

    x = means[z] + stds[z] * eps

    return x


def linear_means(x_prefix, A, b):
    """
    x_prefix: shape (N, i, d)

    A: shape (M, i, d, d)
    b: shape (M, d)

    returns means of shape (N, M, d)
    """
    return torch.einsum("nid,midk->nmk", x_prefix, A) + b[None, :, :]


def sample_autoreg_gmm(N, params, stds, d, s_length):
    """
    params[0]:
        means for first coordinate GMM
        shape (M, d)

    params[i] for i >= 1:
        linear parameters (A_i, b_i)

        A_i shape (M, i, d, d)
        b_i shape (M, d)

    stds[i]:
        standard deviations for coordinate i
        shape (M, d)
    """

    M = params[0].shape[0]
    device = params[0].device

    X = torch.zeros(N, s_length, d, device=device)

    # First coordinate: ordinary GMM
    X[:, 0, :] = sample_gmm_base(
        N=N,
        means=params[0],
        stds=stds[0],
    )

    # Later coordinates: autoregressive GMM
    for i in range(1, s_length):
        A_i, b_i = params[i]

        means_i = linear_means(
            x_prefix=X[:, :i, :],
            A=A_i,
            b=b_i,
        )

        z = torch.randint(M, size=(N,), device=device)
        eps = torch.randn(N, d, device=device)

        X[:, i, :] = means_i[torch.arange(N), z] + stds[i][z] * eps

    return X