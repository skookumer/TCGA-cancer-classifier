import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from pathlib import Path
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import kornia.augmentation as K

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

# class IdBlock(nn.Module):

#     def __init__(self, F):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(F),
#             nn.ReLU(),
#             nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(F),
#         )
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.net(x) + x)
    
# class ConvBlock(nn.Module):

#     def __init__(self, filters, s=2):
#         super().__init__()

#         F1, F2 = filters

#         self.net = nn.Sequential(
#             nn.Conv2d(F1, F2, kernel_size=3, stride=s, padding=1, bias=False),
#             nn.BatchNorm2d(F2),
#             nn.ReLU(),
#             nn.Conv2d(F2, F2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(F2),
#         )

#         self.shortcut = nn.Sequential(
#             nn.Conv2d(F1, F2, kernel_size=1, stride=s, padding=0, bias=False),
#             nn.BatchNorm2d(F2)
#         )

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.net(x) + self.shortcut(x))

class IdBlock(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.conv1 = nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(F)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(F)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + x)

class ConvBlock(nn.Module):
    def __init__(self, filters, s=2):
        super().__init__()
        F1, F2 = filters
        self.conv1 = nn.Conv2d(F1, F2, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(F2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(F2, F2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.downsample = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + self.downsample(x))


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

        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.weight_path = WEIGHTS_PATH / f"{name}.pt"
        if self.weight_path.exists():
            self.net.load_state_dict(torch.load(self.weight_path))
        
    def forward(self, x):
        x = self.net(x)
        x = self.projection(x)
        return F.normalize(x, dim=1), None

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

    def __init__(self, name, num_patches_side, iters, denoise_iter, n_channels, n_classes, levels, patch_dim, contr_dim, conv_image_size, patch_size, dropout, local_consensus_radius=1, consensus_self=True, toprint=True):
        super().__init__()
        
        self.name = name
        self.num_patches_side = num_patches_side
        self.num_patches = self.num_patches_side ** 2
        self.features = []
        self.labels = []
        self.iters = iters
        self.batch_acc = 0
        self.n_levels = levels
        self.denoise_iter = denoise_iter
        self.toprint=toprint

        self.wl =  torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)
        self.wBU = torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)
        self.wTD = torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)
        self.wA =  torch.nn.parameter.Parameter(torch.tensor(0.25), requires_grad=True)

        self.image_to_tokens = nn.Sequential(
            ConvTokenizer_14(in_channels=n_channels, embedding_dim=patch_dim // (patch_size ** 2)),
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

        tokens = self.image_to_tokens(img) #tokenize

        if self.toprint:
            print("embedder output:", tokens.shape)

        n = tokens.shape[1]

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d') #add another dimension to the tensor

        if self.toprint:
            print("bottom_level rearrage", bottom_level.shape)


        '''----loop initialization stuff----'''
        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)
        hiddens = [levels]
        num_contributions = torch.empty(self.n_levels, device=device).fill_(4)
        num_contributions[-1] = 3
        '''----loop initialization stuff----'''


        for _ in range(self.iters):
            levels_with_input = torch.cat((bottom_level, levels), dim=-2)

            if self.toprint:
                print(f"expand tensor for n_levels = {self.n_levels}:", levels_with_input.shape)

            last_level_excluded = levels_with_input[..., :-1, :]

            if self.toprint:
                print("last_level_excluded", last_level_excluded.shape)

            bottom_up_out = self.bottom_up(last_level_excluded)

            top_down_out = self.top_down(torch.flip(levels_with_input[..., 2:, :], [2]))
            top_down_out = F.pad(torch.flip(top_down_out, [2]), (0, 0, 0, 1), value = 0.)

            consensus = self.attention(levels)

            # levels_sum = torch.stack((
            #     levels * self.wl, \
            #     bottom_up_out * self.wBU, \
            #     top_down_out * self.wTD, \
            #     consensus * self.wA
            # )).sum(dim=0)
            # levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            w = F.softmax(torch.stack([self.wl, self.wBU, self.wTD, self.wA]), dim=0)

            levels_sum = torch.stack((
                levels        * w[0],
                bottom_up_out * w[1],
                top_down_out  * w[2],
                consensus     * w[3]
            )).sum(dim=0)
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            levels = levels_mean
            hiddens.append(levels)
        
        all_levels = torch.stack(hiddens)
        top_level = all_levels[self.denoise_iter, :, :, -1]
        top_level = self.contrastive_head(top_level)
        top_level = F.normalize(top_level, dim=1)

        return top_level, all_levels[-1, 0, :, :, :]
    
    def save(self, optimizer, scheduler, epoch, train_loss, val_loss):
        # full checkpoint for resuming training
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, WEIGHTS_PATH / f"{self.name}_resume.pth")


class ConvTokenizer_14(nn.Module):

    def __init__(self, in_channels=3, embedding_dim=64):
        super().__init__()

        F0 = embedding_dim // 4
        F1 = embedding_dim // 2
        F2 = embedding_dim

        #in, out, kernel, stride, pad

        #interleaved stride of 2 to progressively reduce the image size

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, F0, 3, 2, 1, bias=False), #224 -> 112
            nn.BatchNorm2d(F0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F0, F0, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F0, F1, 3, 2, 1, bias=False), #112 -> 56
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1, F1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1, F2, 3, 2, 1, bias=False), #56 -> 28
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F2, F2, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, 1) #28 -> 14
        )

    def forward(self, x):
        return self.net(x)
    
class ConvTokenizer_7(nn.Module):

    def __init__(self, in_channels=3, embedding_dim=64):
        super().__init__()

        F0 = embedding_dim // 8
        F1 = embedding_dim // 4
        F2 = embedding_dim // 2
        F3 = embedding_dim

        #in, out, kernel, stride, pad

        #interleaved stride of 2 to progressively reduce the image size

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, F0, 3, 2, 1, bias=False), #224 -> 112
            nn.BatchNorm2d(F0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F0, F0, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F0, F1, 3, 2, 1, bias=False), #112 -> 56
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1, F1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1, F2, 3, 2, 1, bias=False), #56 -> 28
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F2, F2, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F2, F3, 3, 2, 1, bias=False), #28 -> 14
            nn.BatchNorm2d(F3),
            nn.ReLU(inplace=True),
            nn.Conv2d(F3, F3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, 1) #14 -> 7
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
            coors = torch.stack(torch.meshgrid(torch.arange(num_patches_side), torch.arange(num_patches_side), indexing="ij")).float()

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
    


class Siren(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    

class IMG_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            K.RandomCrop((224, 224), padding=32),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            )
        )

    def forward(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return torch.cat([view1, view2], dim=0)
    

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss