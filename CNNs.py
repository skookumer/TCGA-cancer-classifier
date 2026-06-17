import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from pathlib import Path
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

WEIGHTS_PATH = Path(__file__).parent / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)

TOKEN_ATTEND_SELF_VALUE = -5e-4

def load_weights(name):
    model_name = f"{name}.pt"
    model_weights = WEIGHTS_PATH / model_name
    if model_weights.exists():
        return torch.load(model_name)
    return None

def exists(val):
    return val is not None

class IdBlock(nn.Module):

    def __init__(self, F):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(),
            nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)
    
class ConvBlock(nn.Module):

    def __init__(self, filters, s=2):
        super().__init__()

        F1, F2 = filters

        self.net = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F2)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class Encoder_18(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            IdBlock(64),
            IdBlock(64),
            ConvBlock(filters=[64, 128]),
            IdBlock(128),
            ConvBlock(filters=[128, 256]),
            IdBlock(256),
            ConvBlock(filters=[256, 512]),
            IdBlock(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.weight_path = WEIGHTS_PATH / f"{name}.pt"
        if self.weight_path.exists():
            self.net.load_state_dict(self.weight_path)
        
    def forward(self, x):
        return self.net(x)

    def save(self):
        torch.save(self.net.state_dict(), self.weight_path)




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


class Agglomerator(nn.Module):

    def __init__(self, num_patches_side, iters, denoise_iter, n_channels, n_classes, levels, patch_dim, contr_dim, conv_image_size, patch_size, dropout, local_consensus_radius=1, consensus_self=True):
        super().__init__()
        
        self.num_patches_side = num_patches_side
        self.num_patches = self.num_patches_side ** 2
        self.features = []
        self.labels = []
        self.iters = iters
        self.batch_acc = 0
        self.n_levels = levels
        self.denoise_iter = denoise_iter

        self.wl =  torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)
        self.wBU = torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)
        self.wTD = torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)
        self.wA =  torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)

        self.image_to_tokens = nn.Sequential(
            ConvTokenizer(in_channels=n_channels, embedding_dim=patch_dim // (patch_size ** 2)),
            Rearrange('b d (h p1) (w p2) -> b (h w) (d p1 p2)', p1=patch_size, p2=patch_size)
        )

        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Dropout(p=dropout),
            Rearrange('b n d -> b (n d)'),
            nn.LayerNorm(self.num_patches * patch_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.num_patches * patch_dim, self.num_patches * patch_dim),
            nn.LayerNorm(self.num_patches * patch_dim),
            nn.GELU(),
            nn.LayerNorm(self.num_patches * patch_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.num_patches * patch_dim, contr_dim)
        )

        self.classification_head_from_contr = nn.Sequential(
            nn.Linear(contr_dim, contr_dim),
            nn.GELU(),
            nn.Linear(contr_dim, n_classes)
        )

        self.init_levels = nn.Parameter(torch.randn(self.n_levels, patch_dim))
        self.bottom_up =   ColumnNet(conv_image_size, patch_size, dim=patch_dim, activation=nn.GELU, groups=self.n_levels)
        self.top_down =    ColumnNet(conv_image_size, patch_size, dim=patch_dim, activation=Siren, groups=self.n_levels-1)
        self.attention =   ConsensusAttention(num_patches_side, attend_self=consensus_self, radius=local_consensus_radius)

    def forward(self, img, levels=None):
        b, device = img.shape[0], img.device

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)

        hiddens = [levels]

        num_contributions = torch.empty(self.n_levels, device=device).fill_(4)
        num_contributions[-1] = 3

        for _ in range(self.iters):
            levels_with_input = torch.cat((bottom_level, levels), dim=-2)

            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])

            top_down_out = self.top_down(torch.flip(levels_with_input[..., 2:, :], [2]))
            top_down_out = F.pad(torch.flip(top_down_out, [2]), (0, 0, 0, 1), value = 0.)

            consensus = self.attention(levels)

            levels_sum = torch.stack((
                levels * self.wl, \
                bottom_up_out * self.wBU, \
                top_down_out * self.wTD, \
                consensus * self.wA
            )).sum(dim=0)
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            levels = levels_mean
            hiddens.append(levels)
        
        all_levels = torch.stack(hiddens)
        top_level = all_levels[self.denoise_iter, :, :, -1]
        top_level = self.contrastive_head(top_level)
        top_level = F.normalize(top_level, dim=1)

        return top_level, all_levels[-1, 0, :, :, :]


class ConvTokenizer(nn.Module):

    def __init__(self, in_channels=3, embedding_dim=128):
        super().__init__()

        F0 = embedding_dim // 4
        F1 = embedding_dim // 2
        F2 = embedding_dim

        #in, out, kernel, stride, pad

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, F0, 3, 2, 1, bias=False),
            nn.BatchNorm2d(F0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F0, F0, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F0, F1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1, F1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1, F2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F2, F2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

class ColumnNet(nn.Module):

    def __init__(self, conv_image_size, patch_size, dim, groups, mult=4, activation=nn.GELU):
        super().__init__()

        total_dim = dim * groups
        num_patches = (conv_image_size // patch_size) ** 2

        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.LayerNorm(num_patches),
            nn.Conv1d(total_dim, total_dim * mult, 1, groups=groups),
            activation(),
            nn.LayerNorm(num_patches),
            nn.Conv1d(total_dim * mult, total_dim, 1, groups=groups),
            Rearrange('b (l d) n -> b n l d', l = groups)
        )
    
    def forward(self, levels):
        levels = self.net(levels)
        return levels
    
class ConsensusAttention(nn.Module):

    def __init__(self, num_patches_side, attend_self=True, radius=0):
        super().__init__()

        self.attend_self = attend_self
        self.radius = radius

        if self.radius > 0:
            coors = torch.stack(torch.meshgrid(torch.arange(num_patches_side), torch.arange(num_patches_side))).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)
    
    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim=-1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device=device, dtype=torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out