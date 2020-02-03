import sys, os
import numpy as np
import torch.utils.data as data
import torch
from torchvision import datasets, transforms
from .paired_random_erasing import PairedRandomErasing
from .paired_global_noise import PairedGlobalNoise
from PIL import Image

class Dataset(datasets.MNIST):
    """
    Simulate paird handwritten two-modality data
    """
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--root',
            default='./mnist_data',
            help="the root directory for MNIST dataset"
        )
        parser.add_argument(
            '--modality',
            nargs='+',
            default='X1',
            help="the data modality"
        )
        parser.add_argument(
            '--X2_rotation',
            default=90,
            type=int,
            help="the rotation for second modality"
        )        
        parser.add_argument(
            '--paired_noise',
            default='global_noise',
            help="the paired noise"
        )
        return parser, set()

    def __init__(self, opt, train=True, transform=None, target_transform=None, download=False):
        root = opt.root
        super(Dataset, self).__init__(opt.root, train, transform, target_transform, download)
        self.X2_rotation = opt.X2_rotation
        # compose X2 transformation    
        self.X2_transform = transforms.Compose([transforms.RandomRotation(degrees=[self.X2_rotation, self.X2_rotation])
                                               ])
        # compose noise transform
        #self.simulation_transform = transforms.Compose([transforms.RandomErasing(p=0.7, scale=(0.1, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)]) 
        if opt.paired_noise == 'global_noise':
            # ver 1
            #self.simulation_transform = PairedGlobalNoise(p=1.0, loc=[-0.005, 0.005], scale=[0.005, 0.01], inplace=False)
            # ver 2
            self.simulation_transform = PairedGlobalNoise(p=1.0, loc=[-0.015, 0.0015], scale=[0.005, 0.015], inplace=False)
        elif opt.paired_noise == 'random_erasing':
            self.simulation_transform = PairedRandomErasing(p=1.0, scale=(0.05, 0.15), ratio=(0.3, 3.3), value='random', inplace=False)
        else:
            self.simulation_transform = None          

    def __getitem__(self, index):
        # overwrite the __getitem__ method
        img, target = self.data[index], int(self.targets[index])
        # to return a PIL Image
        img_X1 = Image.fromarray(img.numpy(), mode='L')
        img_X2 = self.X2_transform(img_X1)
        
        if self.simulation_transform is not None: # note the img_X1 and img_X2 are tensors
            img_X1 = transforms.functional.to_tensor(img_X1) # the original image is between 0 and 1
            img_X2 = transforms.functional.to_tensor(img_X2)
            img_X1, img_X2 = self.simulation_transform(img_X1, img_X2)
            img_X1 = transforms.functional.to_pil_image(img_X1)
            img_X2 = transforms.functional.to_pil_image(img_X2)

        # apply transformation to both img_X1 and img_X2
        if self.transform is not None:
            img_X1 = self.transform(img_X1)
            img_X2 = self.transform(img_X2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #print(torch.min(img_X1), torch.max(img_X1))  # min -0.4242, max 2.8215
           
        data = {'labels': target, 
                'X1': img_X1,
                'X2': img_X2}
        return data
