import torch
import torch.nn as nn
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


def load_weights(name):
    model_name = f"{name}.pt"
    model_weights = WEIGHTS_PATH / model_name
    if model_weights.exists():
        return torch.load(model_name)
    return None

class IdBlock(nn.Module):

    def __init__(self, f, filters):
        super().__init__()

        F1, F2, F3 = filters

        self.net = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F3, kernel_size=f, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(F3),
            nn.ReLU(),
            nn.Conv2d(F3, F3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)
    
class ConvBlock(nn.Module):

    def __init__(self, f, filters, s=2):
        super().__init__()

        F1, F2, F3 = filters

        self.net = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F3, kernel_size=f, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(F3),
            nn.ReLU(),
            nn.Conv2d(F3, F3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(F1, F3, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F3)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class Encoder_18(nn.Module):

    def __init__(self, name):
        pass




class Decoder(nn.Module):

    def __init__(self, name, latent_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        weights = load_weights(name)
        if weights is not None:
            self.decoder.load_state_dict(weights)

    def forward(self, x):
        return self.decoder(x)
        

    def masked_mse_loss():
        pass