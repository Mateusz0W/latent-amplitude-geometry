from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from amplitudes.kinematics import vector_to_t
from analysis.latent import (
    CLASS_NAMES,
    decode_z,
    fit_pca_2d,
    transform_pca_2d,
)
from models.vae import VAE


def interpolate_latent(z1: np.ndarray, z2: np.ndarray, n: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """z(α) = (1-α) z1 + α z2, α ∈ [0, 1]."""
    z1 = np.asarray(z1, dtype=np.float32).reshape(-1)
    z2 = np.asarray(z2, dtype=np.float32).reshape(-1)
    alphas = np.linspace(0.0, 1.0, n, dtype=np.float32)
    z = (1.0 - alphas)[:, None] * z1[None, :] + alphas[:, None] * z2[None, :]
    return alphas, z


def decode_trajectory(model: VAE, z_path: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    return decode_z(model, z_path, device=device)


def plot_latent_trajectory(
    mu: np.ndarray,
    y_class: np.ndarray,
    z_path: np.ndarray,
    output_path: str | Path,
) -> None:
    mean, components = fit_pca_2d(mu)
    xy = transform_pca_2d(mu, mean, components)
    path_xy = transform_pca_2d(z_path, mean, components)

    fig, ax = plt.subplots(figsize=(8, 6))
    for class_id, name in enumerate(CLASS_NAMES):
        mask = y_class == class_id
        ax.scatter(xy[mask, 0], xy[mask, 1], alpha=0.35, s=16, label=name)
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="black", lw=2, label="interpolation")
    ax.scatter(path_xy[0, 0], path_xy[0, 1], c="black", s=40, marker="o")
    ax.scatter(path_xy[-1, 0], path_xy[-1, 1], c="black", s=40, marker="s")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent interpolation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_decoded_trajectory(
    s_grid: np.ndarray,
    x_hat: np.ndarray,
    alphas: np.ndarray,
    output_path: str | Path,
) -> None:
    s = np.asarray(s_grid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for vec, alpha in zip(x_hat, alphas):
        t = vector_to_t(vec)
        label = rf"$\alpha={alpha:.2f}$"
        axes[0].plot(s, np.real(t), label=label, alpha=0.85)
        axes[1].plot(s, np.imag(t), label=label, alpha=0.85)
        axes[2].plot(s, np.abs(t) ** 2, label=label, alpha=0.85)

    axes[0].set_title(r"Re$(T)$")
    axes[1].set_title(r"Im$(T)$")
    axes[2].set_title(r"$|T|^2$")
    for ax in axes:
        ax.set_xlabel("s")
        ax.grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, loc="best")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
