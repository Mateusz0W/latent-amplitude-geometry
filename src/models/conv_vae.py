from torch import nn
import torch
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, in_channels: int, img_size: int, hidden_dim: list[tuple[int, int, int, int]], latent_dim: int, encoder_activation: type[nn.Module] = nn.ReLU, decoder_activation: type[nn.Module] = nn.ReLU):
        super().__init__()

        self.encoder = self._build_encoder(in_channels, hidden_dim, encoder_activation)

        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        with torch.no_grad():
            dummy_out = self.encoder(dummy_input)

        self.conv_shape = dummy_out.shape[1:] 
        self.flattened_size = int(torch.prod(torch.tensor(self.conv_shape)))

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = self._build_decoder(hidden_dim, in_channels, decoder_activation)


    def reparameterize(self, logvar, mu):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(logvar, mu)
        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(-1, *self.conv_shape)

        x_recon = self.decoder(z_reshaped)

        return x_recon, mu, logvar

    @staticmethod
    def vae_loss(x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        return recon_loss + kl_loss, recon_loss, kl_loss

    def _build_encoder(self, in_channels: int, hidden_dim: list[tuple[int, int, int, int]], activation: type[nn.Module]):
        layers = []
        in_d = in_channels

        for h_d in hidden_dim:
            layers.append(nn.Conv2d(in_d, h_d[0], kernel_size=h_d[1], stride=h_d[2], padding=h_d[3]))
            layers.append(activation())
            in_d = h_d[0]

        return nn.Sequential(*layers)

    def _build_decoder(self, hidden_dim: list[tuple[int, int, int, int]], out_channels: int, activation: type[nn.Module]):
        layers = []
        in_d = hidden_dim[-1][0]

        reversed_hidden = list(reversed(hidden_dim))

        for idx, h_d in enumerate(reversed_hidden):
            out_ch = reversed_hidden[idx + 1][0] if idx < len(reversed_hidden) - 1 else out_channels
            layers.append(nn.ConvTranspose2d(in_d, out_ch, kernel_size=h_d[1], stride=h_d[2], padding=h_d[3]))

            if idx == len(hidden_dim) - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(activation())

            in_d = out_ch

        return nn.Sequential(*layers)

