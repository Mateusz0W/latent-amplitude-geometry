from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from amplitudes.kinematics import vector_to_t
from models.vae import VAE

CLASS_NAMES = ["K-matrix", "Breit-Wigner", "Mix", "N/D"]
KMATRIX_CLASS = 0


def load_vae(checkpoint_path: str | Path, device: torch.device | None = None) -> VAE:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    hidden_dim = []
    idx = 0
    while f"encoder.{idx}.weight" in state_dict:
        hidden_dim.append(int(state_dict[f"encoder.{idx}.weight"].shape[0]))
        idx += 2

    model = VAE(
        input_dim=int(state_dict["encoder.0.weight"].shape[1]),
        hidden_dim=hidden_dim,
        latent_dim=int(state_dict["fc_mu.weight"].shape[0]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_mu(model: VAE, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    mu, _logvar = model.encode(x)
    return mu.cpu().numpy()


@torch.no_grad()
def decode_z(model: VAE, z: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    x_recon = model.decode(z_t)
    return x_recon.cpu().numpy()


def fit_pca_2d(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = mu.mean(axis=0)
    centered = mu - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:2]


def transform_pca_2d(mu: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (mu - mean) @ components.T


def scatter_latent_2d(
    mu: np.ndarray,
    color: np.ndarray,
    output_path: str | Path,
    title: str,
    colorbar_label: str | None = None,
    cmap: str = "tab10",
    class_names: list[str] | None = None,
) -> None:
    mean, components = fit_pca_2d(mu)
    xy = transform_pca_2d(mu, mean, components)

    fig, ax = plt.subplots(figsize=(8, 6))
    if class_names is not None:
        for class_id, name in enumerate(class_names):
            mask = np.asarray(color) == class_id
            ax.scatter(xy[mask, 0], xy[mask, 1], alpha=0.75, s=18, label=name)
        ax.legend()
    else:
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap=cmap, alpha=0.75, s=18)
        fig.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def kmatrix_neighbor_distance(
    mu: np.ndarray,
    y_class: np.ndarray,
    k: int = 5,
    k_label: int = KMATRIX_CLASS,
) -> np.ndarray:
    """Mean Euclidean distance in latent space to k nearest K-matrix points."""
    y_class = np.asarray(y_class)
    k_idx = np.where(y_class == k_label)[0]
    if k_idx.size == 0:
        raise ValueError("No K-matrix samples in y_class")

    mu_k = mu[k_idx]
    distances = np.linalg.norm(mu[:, None, :] - mu_k[None, :, :], axis=-1)
    distances.sort(axis=1)

    is_k = y_class == k_label
    n_available = mu_k.shape[0]
    k_other = min(k, max(n_available - 1, 1))
    k_all = min(k, n_available)

    result = np.empty(len(mu), dtype=np.float64)
    result[is_k] = distances[is_k, 1 : k_other + 1].mean(axis=1)
    result[~is_k] = distances[~is_k, :k_all].mean(axis=1)
    return result


def plot_kmatrix_distance(
    y_lambda: np.ndarray,
    distances: np.ndarray,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_lambda, distances, alpha=0.6, s=18)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Mean distance to nearest K-matrix (latent)")
    ax.set_title("Distance to unitary manifold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reconstructions(
    model: VAE,
    X: np.ndarray,
    s_grid: np.ndarray,
    y_class: np.ndarray,
    output_path: str | Path,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = next(model.parameters()).device

    indices = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.where(y_class == class_id)[0]
        if candidates.size == 0:
            continue
        indices.append(int(candidates[0]))

    x = torch.from_numpy(np.asarray(X[indices], dtype=np.float32)).to(device)
    with torch.no_grad():
        mu, _logvar = model.encode(x)
        x_recon = model.decode(mu)

    x_np = x.cpu().numpy()
    recon_np = x_recon.cpu().numpy()
    s = np.asarray(s_grid)
    n = len(indices)

    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)
    for row, idx in enumerate(indices):
        t = vector_to_t(x_np[row])
        t_hat = vector_to_t(recon_np[row])
        label = CLASS_NAMES[int(y_class[idx])]
        axes[row, 0].plot(s, np.real(t), label="true")
        axes[row, 0].plot(s, np.real(t_hat), "--", label="recon")
        axes[row, 0].set_title(rf"Re$(T)$ — {label}")
        axes[row, 1].plot(s, np.imag(t), label="true")
        axes[row, 1].plot(s, np.imag(t_hat), "--", label="recon")
        axes[row, 1].set_title(rf"Im$(T)$ — {label}")
        axes[row, 2].plot(s, np.abs(t) ** 2, label="true")
        axes[row, 2].plot(s, np.abs(t_hat) ** 2, "--", label="recon")
        axes[row, 2].set_title(rf"$|T|^2$ — {label}")
        for ax in axes[row]:
            ax.set_xlabel("s")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(
    train_loss: list[float],
    val_loss: list[float],
    output_path: str | Path,
) -> None:
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("VAE loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
