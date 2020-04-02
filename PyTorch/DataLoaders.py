
import math
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data_loader(image_type,  **kwargs):

    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """
    # Generic Transforms: Append these into transforms.
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

    # get training and test directories
    # resize and normalize the images
    trnfms = transforms.Compose([
        transforms.Resize([ kwargs['exp_params']['lr_imageSize'], kwargs['exp_params']['lr_imageSize'] ]),

        #Takes a PIL Image and Scales it between 0 to 1.
        transforms.ToTensor()
        
        #If you want ImageNet initialization
        ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder( 'path/to/dataset/', trnfms)

    #If you want data to split into test and training: Mention how must goes into test
    test_split = int( 20* len(dataset) / 100)

    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-test_split, test_split])
    

    train_loader = DataLoader(dataset=train_set
    						,batch_size=16 , # can be any number
                            ,shuffle = True, #If you want to shuffle the data
                            ,num_workers=0
                            ,drop_last=True ## Drop the ones who werent part of any batch through one iteration
                            )
    
    ## Similarly
    test_loader = DataLoader(dataset=test_set, batch_size=kwargs['exp_params']['batch_size'],
                             num_workers=kwargs['exp_params']['num_workers'], drop_last=True)


    return train_loader, test_loader

