from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal, kl_divergence
from sklearn.manifold import TSNE
import umap
import ot

from amplitudes.kinematics import vector_to_t
from models.vae import VAE

CLASS_NAMES = ["K-matrix", "Breit-Wigner"]
KMATRIX_CLASS = 0


@torch.no_grad()
def encode_mu(model: VAE, X: np.ndarray, device: torch.device | None = None, batch_size: int = 1024) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    mus = []
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(np.asarray(X[start:start + batch_size], dtype=np.float32)).to(device)
        mu, _logvar = model.encode(x)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


@torch.no_grad()
def decode_z(model: VAE, z: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    x_recon = model.decode(z_t)
    return x_recon.cpu().numpy()

def plot_umap(mu: np.ndarray, color: np.ndarray, output_path: str | Path, n_components: int=2, n_neighbors: int=15, min_dist: float=0.1, metric: str="euclidean", random_state: int=42, class_names: list[str] | None = None):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    z_umap = reducer.fit_transform(np.asarray(mu, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(8, 6))

    if class_names is not None:
        for class_id, name in enumerate(class_names):
            mask = np.asarray(color) == class_id
            ax.scatter(
                z_umap[mask, 0],
                z_umap[mask, 1],
                s=10,
                alpha=0.7,
                label=name,
            )
        ax.legend()
    else:
        scatter = ax.scatter(
            z_umap[:, 0],
            z_umap[:, 1],
            c=color,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Przestrzeń latentna VAE — UMAP")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"UMAP ({output_path.name})")

def plot_tsne(mu: np.ndarray, color: np.ndarray, output_path: str | Path, n_components: int=2, perplexity: float=30, init: str="pca", lr: str="auto", random_state: int=42, class_names: list[str] | None = None):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init=init,
        learning_rate=lr,
        random_state=random_state
    )

    mu_embedded = tsne.fit_transform(np.asarray(mu, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if class_names is not None:
        for class_id, name in enumerate(class_names):
            mask = np.asarray(color) == class_id
            ax.scatter(
                mu_embedded[mask, 0],
                mu_embedded[mask, 1],
                s=10,
                alpha=0.7,
                label=name,
            )
        ax.legend()
    else:
        scatter = ax.scatter(
            mu_embedded[:, 0],
            mu_embedded[:, 1],
            c=color,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("t-SNE wymiar 1")
    ax.set_ylabel("t-SNE wymiar 2")
    ax.set_title("Przestrzeń latentna VAE — t-SNE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"t-SNE ({output_path.name})")

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

def calculate_latent_class_distributions(
        model: VAE,
        X: np.ndarray,
        y_class: np.ndarray,
        subset_size: int,
        num_of_drawings: int,
        device: torch.device | None = None,
        seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:

    PAIR_NAMES = ("Unitary-Unitary", "Nonunitary-Nonunitary", "Mix-Mix", "Unitary-Nonunitary")

    #prepare data
    y_class = np.asarray(y_class)
    model.eval()
    mask_unit = y_class == KMATRIX_CLASS  
    Z = encode_mu(model, X, device)      
    Z_unit = Z[mask_unit]             
    Z_nonunit = Z[~mask_unit]  
    Z_mix = Z

    rng = np.random.default_rng(seed)

    results = {name: {"kl": [], "wasserstein": []} for name in PAIR_NAMES}

    for idx, _ in enumerate(range(num_of_drawings), start=1):
        print(f"\rcalculating latent class distributions: {(idx )/num_of_drawings * 100:.2f}%", end="", flush=True)
        #sampling of subsets
        Z_unit1, Z_unit2 = _select_two_disjoint_subsets(Z_unit, subset_size, subset_size, rng)
        Z_nonunit1, Z_nonunit2 = _select_two_disjoint_subsets(Z_nonunit, subset_size, subset_size, rng)
        Z_mix1, Z_mix2 = _select_two_disjoint_subsets(Z_mix, subset_size, subset_size, rng)
        Z_unitary, Z_nonunitary = _select_subset(Z_unit, subset_size, rng), _select_subset(Z_nonunit, subset_size, rng)

        #calculate metrics
        for metric in ["kl", "wasserstein"]:
            f = _calculate_kl_divergence if metric == "kl" else _calculate_wasserstein_distance
            results["Unitary-Unitary"][metric].append(f(Z_unit1, Z_unit2))
            results["Nonunitary-Nonunitary"][metric].append(f(Z_nonunit1, Z_nonunit2))
            results["Mix-Mix"][metric].append(f(Z_mix1, Z_mix2))
            results["Unitary-Nonunitary"][metric].append(f(Z_unitary, Z_nonunitary))

    print()
    return {name: {m: np.asarray(v) for m, v in metrics.items()} for name, metrics in results.items()}
 
def _select_two_disjoint_subsets(data: np.ndarray, subset1_size: int, subset2_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    assert subset1_size + subset2_size <= len(data), "The sizes of the subsets exceed the size of the set"
    idx = rng.choice(len(data), subset1_size + subset2_size, replace=False)
    return data[idx[:subset1_size]], data[idx[subset1_size:]]

def _select_subset(data: np.ndarray, subset_size: int, rng: np.random.Generator) -> np.ndarray:
    assert subset_size <= len(data), "The size of the subset exceed the size of the set"
    return data[rng.choice(len(data), subset_size, replace=False)]

def _calculate_wasserstein_distance(A: np.ndarray, B: np.ndarray, metric: str="euclidean", numItermax=1_000_000):
    M = ot.dist(A, B, metric=metric)
    return ot.emd2(ot.unif(len(A)), ot.unif(len(B)), M, numItermax=numItermax)

def _fit_gauss(Z):
    Z = torch.as_tensor(Z, dtype=torch.float64)
    cov = torch.cov(Z.T) + 1e-6 * torch.eye(Z.shape[1], dtype=torch.float64)
    return MultivariateNormal(Z.mean(0), covariance_matrix=cov)

def _calculate_kl_divergence(A: np.ndarray, B: np.ndarray):
    p, q = _fit_gauss(A), _fit_gauss(B)
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).item()

def plot_metric_histograms(
    results: dict[str, dict[str, np.ndarray]],
    output_dir: str | Path,
    pairs: list[str] | None = None,
    log_scale: bool = True,
    suffix: str = "",
) -> None:
    df = pd.DataFrame([
            {"pair": pair, "metric": metric, "value": float(v)}
            for pair, metrics in results.items()
            for metric, values in metrics.items()
            for v in values
        ])
    if pairs is not None:
        df = df[df["pair"].isin(pairs)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    METRIC_LABELS = {
        "kl": r"Symetryczna $D_{KL}$",
        "wasserstein": r"Wasserstein $W_1$",
    }

    for metric, label in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            data=df[df["metric"] == metric],
            x="value",
            hue="pair",
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            kde=True,
            log_scale=log_scale,
            ax=ax,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Gęstość")
        ax.set_title(f"{label} między podzbiorami w przestrzeni latentnej")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = output_dir / f"hist_{metric}{suffix}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Histogram ({out_file.name})")

