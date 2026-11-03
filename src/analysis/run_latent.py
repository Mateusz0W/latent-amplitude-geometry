import argparse
import shutil
from pathlib import Path

import numpy as np

from analysis.latent import (
    CLASS_NAMES,
    encode_mu,
    kmatrix_neighbor_distance,
    plot_kmatrix_distance,
    plot_reconstructions,
    plot_umap,
    plot_tsne,
    plot_pca,
)

from utils.utils import load_vae, get_src_and_repo_path, resolve_path

_REPO_ROOT = get_src_and_repo_path()[1]


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
    ckpt_path = resolve_path(checkpoint)
    npz_path = resolve_path(data_path)
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
    
    plot_pca(mu, y_class, latent_dir / "pca.png", title="Latent PCA — klasa", class_names=CLASS_NAMES, n_components=3)
    plot_umap(mu, y_class, latent_dir / "umap.png", class_names=CLASS_NAMES)    
    plot_tsne(mu, y_class,latent_dir / "t-sne.png", class_names=CLASS_NAMES)

    distances = kmatrix_neighbor_distance(mu, y_class, k=5)
    plot_kmatrix_distance(y_lambda, distances, latent_dir / "kmatrix_distance.png")

    np.savez(
        latent_dir / "mu.npz",
        mu=mu,
        y_class=y_class,
        y_lambda=y_lambda,
        distances=distances,
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
