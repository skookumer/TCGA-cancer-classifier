import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from CNNs import Encoder_18, IdBlock, ConvBlock, Agglomerator
import kornia.augmentation as K

WEIGHTS_PATH = Path(__file__).parent / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)

def K_transform():
    return nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=360),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        )

def K_normalize():
    return K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )


class CLF_HEAD(nn.Module):

    def __init__(self, name, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
            # nn.Softmax(dim=1)
        )

        self.weight_path = WEIGHTS_PATH / f"{name}_clf.pt"
        if self.weight_path.exists():
            self.net.load_state_dict(torch.load(self.weight_path))

    def forward(self, x):
        return self.net(x)

    def save(self):
        torch.save(self.net.state_dict(), self.weight_path)

class PROB_HEAD(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.weight_path = WEIGHTS_PATH / f"{name}_prob.pt"
        if self.weight_path.exists():
            self.net.load_state_dict(torch.load(self.weight_path))

    def forward(self, x):
        return self.net(x).squeeze(1)
    
    def save(self):
        torch.save(self.net.state_dict(), self.weight_path)
    

class RESNET(nn.Module):

    def __init__(self, name, unfreeze_last=False, dropout=0.5):
        super().__init__()

        self.unfreeze_last = unfreeze_last
        self.name = name

        # check for regularization/params
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity() #remove the final linear layer to get 512 vector
        self.pretrained = resnet
        self.last = self.pretrained.layer4

        for param in self.pretrained.parameters():
            param.requires_grad = False

        if unfreeze_last:
            # self.last = nn.Sequential(
            #     ConvBlock(filters=[256, 512]),
            #     nn.Dropout(dropout),
            #     IdBlock(512),
            #     nn.Dropout(dropout)
            # )
            self.last = nn.Sequential(self.pretrained.layer4,
                                      nn.Dropout(dropout))
            self.pretrained.layer4 = self.last

            layer4_path = WEIGHTS_PATH / f"{name}_layer4.pt"
            if layer4_path.exists():
                self.last.load_state_dict(torch.load(layer4_path))
            # else:
            #     pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            #     self.last[0].load_state_dict(pretrained.layer4[0].state_dict())
            #     self.last[2].load_state_dict(pretrained.layer4[1].state_dict())
        
        self.head = CLF_HEAD(name, dropout)
        self.gpu_transform = K_transform()
        self.gpu_normalize = K_normalize()

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x = self.pretrained(x)
        return self.head(x)

    def save(self):
        self.head.save()
        if self.unfreeze_last:
            torch.save(self.last.state_dict(), WEIGHTS_PATH / f"{self.name}_last.pt")

class RESNET_CUSTOM(nn.Module):

    def __init__(self, name="enc_18", unfreeze_last=False, dropout=0.1):
        super().__init__()

        self.unfreeze_last = unfreeze_last
        self.name = name

        # check for regularization/params
        resnet = Encoder_18(self.name)
        self.pretrained = resnet

        for param in self.pretrained.parameters():
            param.requires_grad = False
        
        self.head = CLF_HEAD(name, dropout)
        self.gpu_transform = K_transform()
        self.gpu_normalize = K_normalize()

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x, _ = self.pretrained(x)
        return self.head(x)

    def save(self):
        self.head.save()
        if self.unfreeze_last:
            torch.save(self.last.state_dict(), WEIGHTS_PATH / f"{self.name}.pt")

class INCEPTION(nn.Module):
    def __init__(self, name, unfreeze_last=False, dropout=0.5):
        super().__init__()
        self.unfreeze_last = unfreeze_last
        self.name = name

        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, dropout=dropout)
        inception.avgpool = nn.Identity()
        inception.fc = nn.Identity()
        inception.aux_logits = False
        inception.fc = nn.Identity()
        self.pretrained = inception
        self.last = self.pretrained.Mixed_7c

        for param in self.pretrained.parameters():
            param.requires_grad = False

        if unfreeze_last:
            self.last = nn.Sequential(
                self.pretrained.Mixed_7c,
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.pretrained.Mixed_7c = self.last
            last_path = WEIGHTS_PATH / f"{name}_last.pt"
            if last_path.exists():
                self.last.load_state_dict(torch.load(last_path))
            for param in self.last.parameters():
                param.requires_grad = True

        self.head = CLF_HEAD(name)
        self.gpu_transform = K_transform()
        self.gpu_normalize = K_normalize()

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x = self.pretrained(x)
        return self.head(x)

    def save(self):
        self.head.save()
        if self.unfreeze_last:
            torch.save(self.last.state_dict(), WEIGHTS_PATH / f"{self.name}_last.pt")


class VGG16(nn.Module):
    def __init__(self, name, unfreeze_last=False, dropout=0.5):
        super().__init__()
        self.unfreeze_last = unfreeze_last
        self.name = name

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.classifier[2] = nn.Dropout(dropout)
        vgg.classifier[5] = nn.Dropout(dropout)
        vgg.classifier[6] = nn.Identity()

        features = list(vgg.features.children())
        vgg.features = nn.Sequential(*features[:24])
        vgg.classifier = nn.Identity()
        self.last = nn.Sequential(*features[24:],
                                  nn.Dropout(dropout),
                                  nn.AdaptiveAvgPool2d((1, 1)), #512 dimensions
                                  nn.Flatten())
        self.pretrained = vgg

        for param in self.pretrained.parameters():
            param.requires_grad = False

        if unfreeze_last:
            last_path = WEIGHTS_PATH / f"{name}_last.pt"
            if last_path.exists():
                self.pretrained.features[24:].load_state_dict(torch.load(last_path))  # note: won't work directly
            for param in list(self.pretrained.features.children())[24:]:
                param.requires_grad = True

        self.head = CLF_HEAD(name)
        self.gpu_transform = K_transform()
        self.gpu_normalize = K_normalize()

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x = self.pretrained.features(x)
        x = self.last(x)
        return self.head(x)

    def save(self):
        self.head.save()
        if self.unfreeze_last:
            torch.save(self.last.state_dict(), WEIGHTS_PATH / f"{self.name}_last.pt")


class AGG_14(nn.Module):

    def __init__(self, name="agg_14", dropout=0.1):
        super().__init__()
        self.name = name

        
        self.pretrained = Agglomerator(
            name=name,
            num_patches_side=14,
            iters=4,
            denoise_iter=-1,
            n_channels=3,
            n_classes=10,
            levels=2,
            patch_dim=64,
            contr_dim=512,
            conv_image_size=14,
            patch_size=1,
            dropout=0.3,
            local_consensus_radius=0,
            toprint=False
        )

        weights = torch.load(WEIGHTS_PATH / f"{name}.pth")
        self.pretrained.load_state_dict(weights["model_state_dict"])

        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.head = CLF_HEAD(name, dropout)
        self.gpu_transform = K_transform()
        self.gpu_normalize = K_normalize()

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x, _ = self.pretrained(x)
        return self.head(x)
    
    def save(self):
        self.head.save()
