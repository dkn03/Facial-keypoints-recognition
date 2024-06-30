import torch
from torchvision.models import resnet18, resnet152
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

class ModifiedResnet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 68 * 2)

    def forward(self, x):
        out = self.model(x)
        #return out.reshape(-1, 2)
        return out
    
class ModifiedResnet152(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet152()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 68 * 2)

    def forward(self, x):
        out = self.model(x)
        #return out.reshape(-1, 2)
        return out
    

