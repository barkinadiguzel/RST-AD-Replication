import torch
import torch.nn as nn
import torchvision.models as models

class TeacherResNet(nn.Module):
    def __init__(self, version="resnet18", pretrained=True, feature_levels=["layer1","layer2","layer3"]):
        super().__init__()
        self.feature_levels = feature_levels
        
        if version == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif version == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported teacher version")

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  

    def forward(self, x):
        x = self.initial(x)
        features = []
        if "layer1" in self.feature_levels:
            x1 = self.layer1(x)
            features.append(x1)
        if "layer2" in self.feature_levels:
            x2 = self.layer2(features[-1])
            features.append(x2)
        if "layer3" in self.feature_levels:
            x3 = self.layer3(features[-1])
            features.append(x3)
        if "layer4" in self.feature_levels:
            x4 = self.layer4(features[-1])
            features.append(x4)
        return features
