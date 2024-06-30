import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2  
from PIL import Image

TRANSFORMS = v2.Compose([
    v2.ToTensor(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FaceDataset(Dataset):

    def __init__(self, image_names: list[str], image_path: str, labels: dict[str: list], device=torch.device('cpu'), transforms=None):
        self.transforms = transforms
        self.image_names = image_names
        self.image_path = image_path
        self.labels = labels
        self.device = device


    def __getitem__(self, index):
        
        image = Image.open(self.image_path + self.image_names[index])
        label = self.labels[self.image_names[index]]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        #label = label.to(device=self.device)
        label = label.reshape(1, -1).squeeze()
        if self.transforms:
            image = self.transforms(image)

        return image, label
    
    def __len__(self):
        return len(self.image_names)