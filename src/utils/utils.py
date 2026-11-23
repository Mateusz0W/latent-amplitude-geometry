import yaml
from pathlib import Path
import sys
import torch
from models.vae import VAE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_src_and_repo_path() -> tuple[Path, Path]:
    _src_root = Path(__file__).resolve().parents[1]
    _repo_root = _src_root.parent
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

    return _src_root, _repo_root

def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    _, repo_root = get_src_and_repo_path()
    return repo_root / path
        
def load_config(yaml_path: Path) -> dict:
    yaml_path = resolve_path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_vae(checkpoint_path: str | Path, device: torch.device | None = None) -> VAE:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_path(checkpoint_path)
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

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_NAMES = ("train", "val", "test")

def split_by_physics_id(physics_id: np.ndarray, split_cfg: dict, seed: int) -> np.ndarray:
    """Per-sample split ids; all samples sharing a physics_id land in the same split."""
    train_f = float(split_cfg.get("train", 0.8))
    val_f = float(split_cfg.get("val", 0.1))
    test_f = float(split_cfg.get("test", 0.1))
    total = train_f + val_f + test_f
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")
    if any(f < 0 for f in (train_f, val_f, test_f)):
        raise ValueError("split fractions must be non-negative")

    unique_ids, inverse = np.unique(np.asarray(physics_id), return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique_ids))

    n_train = int(len(unique_ids) * train_f)
    n_val = int(len(unique_ids) * val_f)
    id_split = np.full(len(unique_ids), SPLIT_TEST, dtype=np.int32)
    id_split[order[:n_train]] = SPLIT_TRAIN
    id_split[order[n_train:n_train + n_val]] = SPLIT_VAL
    return id_split[inverse]

def plot_pca(X: np.ndarray, color: np.ndarray, output_path: str | Path, title: str, colorbar_label: str | None = None, cmap: str = "tab10", class_names: list[str] | None = None, n_components: int = 2, standardization: bool = False):
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    pca = PCA(n_components=n_components)
    X = StandardScaler().fit_transform(X) if standardization else X
    xy = pca.fit_transform(np.asarray(X, dtype=np.float64))
    ratios = pca.explained_variance_ratio_
    total = float(ratios.sum())
    result_str = "".join(
        f"PC{idx}={100 * ratio:.1f}% " for idx, ratio in enumerate(ratios, start=1)
    )
    var_label = f"{result_str}(łącznie {100 * total:.1f}%)"

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    if class_names is not None:
        for class_id, name in enumerate(class_names):
            mask = np.asarray(color) == class_id
            if n_components == 2:
                ax.scatter(xy[mask, 0], xy[mask, 1], alpha=0.75, s=18, label=name)
            else:
                ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2], alpha=0.75, s=18, label=name)
        ax.legend()
    else:
        if n_components == 2:
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap=cmap, alpha=0.75, s=18)
        else:
            scatter = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=color, cmap=cmap, alpha=0.75, s=18)
        fig.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if n_components == 3:
        ax.set_zlabel("PC3")

    ax.set_title(f"{title}\n{var_label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"PCA ({Path(output_path).name}): {var_label}")
    