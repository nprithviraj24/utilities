##Transforming a dataset:
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import torch
from torch import nn
import torchvision.datasets as datasets

class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05):
        super().__init__()
        self.noise = torch.zeros(shape, shape)
        self.std = std

    def forward(self, x):
        x = torchvision.transforms.functional.to_tensor(x)
        if not self.training: return x
        self.noise.data.normal_(0, std=self.std)

        print(x.size(), self.noise.size())
        x = x + self.noise
        return torchvision.transforms.functional.to_pil_image(x)


data_dir = "data-dir"
save_dir = "save_dir"

trnfms = tfs.Compose([
    tfs.Resize(256),

    tfs.CenterCrop(130),

    DynamicGNoise(130),

    tfs.Resize(8),

    tfs.ToTensor()
    ])

data = datasets.ImageFolder(data_dir, trnfms)

loader = DataLoader(dataset=data, batch_size=1, shuffle=True, drop_last=False)

for btch, (x,_) in tqdm(enumerate(loader), total=len(loader)):

    vutils.save_image(x, save_dir + str(btch) + ".png")

