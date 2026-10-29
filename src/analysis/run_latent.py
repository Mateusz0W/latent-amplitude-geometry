import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from analysis.latent import (
    CLASS_NAMES,
    encode_mu,
    kmatrix_neighbor_distance,
    load_vae,
    plot_kmatrix_distance,
    plot_reconstructions,
    scatter_latent_2d,
    plot_umap,
    plot_tsne,
)
from analysis.trajectories import (
    decode_trajectory,
    interpolate_latent,
    plot_decoded_trajectory,
    plot_latent_trajectory,
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def default_checkpoint() -> Path:
    candidates = [
        _REPO_ROOT / "src" / "models" / "best_vae.pt",
        _REPO_ROOT / "results" / "vae" / "best_vae.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def run(checkpoint: str | Path, data_path: str | Path) -> None:
    ckpt_path = _resolve_path(checkpoint)
    npz_path = _resolve_path(data_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Nie znaleziono datasetu: {npz_path}")

    with np.load(npz_path) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y_class = np.asarray(data["y_class"])
        y_lambda = np.asarray(data["y_lambda"])
        s_grid = np.asarray(data["s_grid"])

    model = load_vae(ckpt_path)
    mu = encode_mu(model, X)

    vae_dir = _REPO_ROOT / "results" / "vae"
    latent_dir = _REPO_ROOT / "results" / "latent"
    vae_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, vae_dir / "best_vae.pt")

    plot_reconstructions(model, X, s_grid, y_class, vae_dir / "reconstructions.png")
    scatter_latent_2d(
        mu,
        y_class,
        latent_dir / "scatter_class.png",
        title="Latent PCA — klasa",
        class_names=CLASS_NAMES,
    )
    scatter_latent_2d(
        mu,
        y_lambda,
        latent_dir / "scatter_lambda.png",
        title=r"Latent PCA — $\lambda$",
        colorbar_label=r"$\lambda$",
        cmap="viridis",
    )

    plot_umap(mu, y_class, latent_dir / "umap.png", class_names=CLASS_NAMES)    
    plot_tsne(mu, y_class,latent_dir / "t-sne.png", class_names=CLASS_NAMES)

    distances = kmatrix_neighbor_distance(mu, y_class, k=5)
    plot_kmatrix_distance(y_lambda, distances, latent_dir / "kmatrix_distance.png")

    rng = np.random.default_rng(42)
    k_idx = rng.choice(np.where(y_class == 0)[0])
    bw_idx = rng.choice(np.where(y_class == 1)[0])
    alphas, z_path = interpolate_latent(mu[k_idx], mu[bw_idx], n=21)
    x_hat = decode_trajectory(model, z_path)
    plot_latent_trajectory(mu, y_class, z_path, latent_dir / "trajectory_scatter.png")
    plot_decoded_trajectory(s_grid, x_hat, alphas, latent_dir / "trajectory_decoded.png")

    np.savez(
        latent_dir / "mu.npz",
        mu=mu,
        y_class=y_class,
        y_lambda=y_lambda,
        distances=distances,
        k_idx=np.asarray([k_idx]),
        bw_idx=np.asarray([bw_idx]),
    )
    print(f"Zapisano rekonstrukcje: {vae_dir / 'reconstructions.png'}")
    print(f"Zapisano figury latent: {latent_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Krok 6–7: rekonstrukcje VAE i narzędzia latent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint()),
        help="Ścieżka do best_vae.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/amplitudes.npz",
        help="Ścieżka do datasetu .npz",
    )
    args = parser.parse_args()
    run(args.checkpoint, args.data)


if __name__ == "__main__":
    main()
