import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from models.vae import VAE
from early_stopping import EarlyStopping

def split_data(data: np.ndarray, batch_size: int=32, train_size: int=0.7, test_size: int=0.15) -> tuple[DataLoader, DataLoader, DataLoader]:
    if (train_size + test_size) >= 1:
        raise ValueError("The size of the sets cannot exceed 1")
    if any(x < 0 for x in [train_size, test_size]):
        raise ValueError("The sizes of the sets must be positive")

    train_len = int(len(data) * train_size)
    test_len = int(len(data) * test_size)
    val_len = len(data) - train_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(
        data,
        [train_len, val_len, test_len]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def validate_model(model, val_loader):
    model.eval()
    
    val_loss = 0.0
    val_recon = 0.0
    val_kl = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            x_recon, mu, logvar = model(x)

            loss, recon_loss, kl_loss = model.vae_loss(
                x_recon, x, mu, logvar
            )

            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)

    return val_loss, val_recon, val_kl


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(_SRC_ROOT.parent / "data" / "amplitudes.npz")
    X = np.asarray(data["X"], dtype=np.float32)

    train_loader, val_loader, test_loader = split_data(X)

    model = VAE(
        input_dim=X.shape[1],
        hidden_dim=[128, 64],
        latent_dim=16
    ).to(device)

    early_stopping = EarlyStopping()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for x in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            x_recon, mu, logvar = model(x)

            loss, recon_loss, kl_loss = model.vae_loss(
                x_recon, x, mu, logvar
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        val_loss, val_recon, val_kl = validate_model(model, val_loader)
        early_stopping(val_loss, model)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"(Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
            f"Val Loss: {val_loss:.4f} "
            f"(Recon: {val_recon:.4f}, KL: {val_kl:.4f})"
        )

        if early_stopping.early_stop:
            print("Early Stopping Triggerd")
            break

    if early_stopping.best_model is not None:
        model.load_state_dict(early_stopping.best_model)
        model_path = _SRC_ROOT.parent / "src" / "models" / "best_vae.pt"
        torch.save(model.state_dict(), model_path)

        print(f"Best model saved to: {model_path}")