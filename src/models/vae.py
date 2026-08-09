import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def  __init__(self, input_dim: int, hidden_dim: list[int], latent_dim: int, encoder_activation: type[nn.Module] = nn.ReLU, decoder_activation: type[nn.Module] = nn.ReLU):
        super().__init__()

        self.encoder = self._build_encoder(input_dim, hidden_dim, encoder_activation)
        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim[-1], latent_dim)
        self.decoder = self._build_decoder(latent_dim, hidden_dim, input_dim, decoder_activation)

    def reparameterize(self, logvar, mu):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(logvar, mu)

        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def vae_loss(x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        return recon_loss + kl_loss, recon_loss, kl_loss

    def _build_encoder(self, input_dim: int, hidden_dim: list[int], activation: type[nn.Module]):
        layers = []
        in_d = input_dim

        for h_d in hidden_dim:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(activation())
            in_d = h_d

        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim: int, hidden_dim: list[int], output_dim: int, activation: type[nn.Module]):
        layers = []
        in_d = latent_dim

        for h_d in reversed(hidden_dim):
            layers.append(nn.Linear(in_d, h_d))
            layers.append(activation())
            in_d = h_d

        layers.append(nn.Linear(in_d, output_dim))

        return nn.Sequential(*layers)
    

