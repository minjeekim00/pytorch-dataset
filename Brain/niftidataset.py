import torch
import torch.utils.data as data
import nibabel as nib
import os
import os.path
import sys
import numpy as np

def make_dataset(dir):
    data = []
    
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            item = path
            data.append(item)
                    
    return data
 

def nii_loader(path):
    data = nib.load(path)
    image = np.array(data.get_fdata(), dtype=np.float32)
    image = np.transpose(image, (2, 1, 0))
    return image



class NiftiDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/.nii.gz

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
            
     Attributes:
        samples (list): List of sample path
    """

    def __init__(self, root, labels, loader=nii_loader, transform=None, target_transform=None, preprocess=True):
        samples = make_dataset(root)
        
        if len(samples[0]) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root ))

        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, label) where label is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        
        if self.preprocess:
            sample = self.preprocess_img(sample)
            
        if self.transform is not None:
            sample = self.transform(sample)
        sample = self.to_tensor(sample)
        return sample
   
    def __getpath__(self, index):
        path = self.samples[index]
        return path

    def __len__(self):
        return len(self.samples[0])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    
    def preprocess_img(self, img):
        ### TODO:
        return img
        
    def to_tensor(self, img):
        return torch.from_numpy(img.transpose((3, 0, 1, 2)))

    

if __name__ == "__main__":
    train_img_dataset = NiftiDatasetFolder('/data/project/rw/ABCD_Challenge/training/image')
    train_mask_dataset = NiftiDatasetFolder('/data/project/rw/ABCD_Challenge/training/mask')
