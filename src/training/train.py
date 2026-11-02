from pathlib import Path
import argparse

import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim

from models.vae import VAE
from training.early_stopping import EarlyStopping
from utils.utils import resolve_path, load_config

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

def validate_model(model, val_loader, device):
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config (relative to repo root or absolute)",
    )
    args = parser.parse_args()
    yaml_path = Path(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(resolve_path("data/amplitudes.npz"))
    X = np.asarray(data["X"], dtype=np.float32)
    y_class = np.asarray(data["y_class"])
    s_grid = np.asarray(data["s_grid"])

    config = load_config(yaml_path)["vae"]
    train_loader, val_loader, test_loader = split_data(X, batch_size=config["batch_size"], train_size=config["train_size"], test_size=config["test_size"])

    model = VAE(
        input_dim=X.shape[1],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"]
    ).to(device)

    early_stopping = EarlyStopping(patience=config["patience"], min_delta= config["min_delta"])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    num_epochs = config["num_epochs"]
    train_losses = []
    val_losses = []

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

        val_loss, val_recon, val_kl = validate_model(model, val_loader, device)
        early_stopping(val_loss, model)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"(Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
            f"Val Loss: {val_loss:.4f} "
            f"(Recon: {val_recon:.4f}, KL: {val_kl:.4f})"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if early_stopping.early_stop:
            print("Early Stopping Triggerd")
            break

    if early_stopping.best_model is not None:
        model.load_state_dict(early_stopping.best_model)
        model_path = resolve_path("src/models/best_vae.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved to: {model_path}")

        from analysis.latent import plot_loss_curves, plot_reconstructions

        results_dir = resolve_path("results/vae")
        results_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), results_dir / "best_vae.pt")
        plot_loss_curves(train_losses, val_losses, results_dir / "loss_curves.png")
        plot_reconstructions(
            model, X, s_grid, y_class, results_dir / "reconstructions.png"
        )
        print(f"Zapisano figury VAE: {results_dir}")


if __name__ == "__main__":
    main()