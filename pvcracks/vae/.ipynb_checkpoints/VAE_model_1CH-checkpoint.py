import torch.nn as nn
from termcolor import colored
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
			nn.Conv2d(512, 1024, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1024 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 7 * 7, latent_dim)
        print(colored(self.fc_mu, 'green'))
        print(colored(self.fc_logvar, 'blue'))
    
    def forward(self, x):
        x = self.conv(x)
        print(colored(x.size(), 'red'))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 7 * 7)# * 25 * 25)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 7, 7)
        x_deconv = self.deconv(x)
        # x_deconv = nn.functional.upsample(x_deconv, size=(400, 400), mode='bilinear', align_corners=False)
        return x_deconv

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar