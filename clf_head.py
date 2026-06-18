import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from CNNs import Encoder_18
import kornia.augmentation as K

WEIGHTS_PATH = Path(__file__).parent / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


class CLF_HEAD(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5), #high dropout from the paper
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

    def __init__(self, name, unfreeze_l4=True):
        super().__init__()

        self.unfreeze_l4 = unfreeze_l4
        self.name = name

        # check for regularization/params
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity() #remove the final linear layer to get 512 vector
        self.resnet = resnet
        for param in self.resnet.parameters():
            param.requires_grad = False

        if unfreeze_l4:
            for param in resnet.layer4.parameters():
                param.requires_grad = True
                layer4_path = WEIGHTS_PATH / f"{name}_layer4.pt"
                if layer4_path.exists():
                    self.resnet.layer4.load_state_dict(torch.load(layer4_path))
        
        self.head = CLF_HEAD(name)
        self.prob = PROB_HEAD(name)

        self.gpu_transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=360),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        )

        self.gpu_normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x = self.resnet(x)
        return self.head(x), self.prob(x)

    def save(self):
        self.head.save()
        self.prob.save()
        if self.unfreeze_l4:
            torch.save(self.resnet.layer4.state_dict(), WEIGHTS_PATH / f"{self.name}_layer4.pt")

class RESNET_custom(nn.Module):

    def __init__(self, classifier_name):
        super().__init__()

        self.resnet = Encoder_18(f"encoder_{classifier_name}")
        self.head = CLF_HEAD(classifier_name)

        self.gpu_transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=360),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        )

        self.gpu_normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

    def forward(self, x):
        if self.training:
            x = self.gpu_transform(x)
        x = self.gpu_normalize(x)
        x = self.resnet(x)
        return self.head(x)

    def save(self):
        self.head.save()
        self.resnet.save()