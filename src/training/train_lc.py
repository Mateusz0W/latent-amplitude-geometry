import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.linear_classifier import LinearClassifier
from early_stopping import EarlyStopping
from analysis.latent import load_vae, encode_mu




def make_loader(z, y, mask, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(z[mask]),
        torch.from_numpy(y[mask]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def validate_model(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return val_loss / len(loader), correct / total

def load_config(yaml_path: str):
    if not yaml_path.is_absolute():
        yaml_path = _REPO_ROOT / yaml_path
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["linear_probing"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear classifier model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config (relative to repo root or absolute)",
    )

    args = parser.parse_args()
    yaml_path = Path(args.config)

    config = load_config(yaml_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(_SRC_ROOT.parent / "data" / "amplitudes.npz")
    X = np.asarray(data["X"], dtype=np.float32)
    vae = load_vae(_REPO_ROOT / config["checkpoint"])  
    z = encode_mu(vae, X)                 
    y = np.asarray(data["y_class"], dtype=np.float32).reshape(-1, 1)
    split = np.asarray(data["split"])   


    batch_size = config["batch_size"]
    train_loader = make_loader(z, y, split == 0, batch_size, shuffle=True)
    val_loader   = make_loader(z, y, split == 1, batch_size, shuffle=False)
    test_loader  = make_loader(z, y, split == 2, batch_size, shuffle=False)

    model = LinearClassifier(z.shape[1]).to(device)
    early_stopping = EarlyStopping(patience=config["patience"])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    epochs = config["num_epochs"]

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(early_stopping.best_model)
    test_loss, test_acc = validate_model(model, test_loader, criterion, device)
    print(f"Test acc: {test_acc:.4f}")



