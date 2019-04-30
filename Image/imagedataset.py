import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import *

import os
import numpy as np
from PIL import Image

from tqdm import tqdm
from glob import glob


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx):
    fnames, labels = [], []
    
    for label in sorted(os.listdir(dir)):
        for fname in sorted(os.listdir(os.path.join(dir, label))):
            for frame in sorted(os.listdir(os.path.join(dir, label, fname))):
                if not is_image_file(frame):
                    continue
                fnames.append(os.path.join(dir, label, fname, frame))
                labels.append(label)
            
    assert len(labels) == len(fnames)
    print('Number of {} images: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    return [fnames, targets]

def labels_to_idx(labels):
    labels_dict = {label: i for i, label in enumerate(sorted(set(labels)))}
    return np.array([labels_dict[label] for label in labels], dtype=int)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageDataset(ImageFolder):
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[0][index]
        target = self.samples[1][index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __getpath__(self, index):
        return self.samples[0][index]
    
    def __len__(self):
        return len(self.samples[0]) # fnames
        