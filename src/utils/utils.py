import yaml
from pathlib import Path
import sys
import torch
from models.vae import VAE

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
    