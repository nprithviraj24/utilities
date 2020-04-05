##Transforming a dataset:

import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import torchvision.datasets as datasets


trnfms = tfs.Compose([
    tfs.Pad((100, 50, 100, 50), fill=0, padding_mode='constant'),
    tfs.Resize([540,540]),
    tfs.ToTensor()
    ])

data = datasets.ImageFolder('/images/directory/', trnfms)

loader = DataLoader(dataset=data, batch_size=1, shuffle=True, drop_last=False)

for btch, (x,_) in tqdm(enumerate(loader), total=len(loader)):

    vutils.save_image(x, "/file/to/be/saved/" + str(btch) + ".png")