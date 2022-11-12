import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


# def image_loader(image_name):
#     if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
#         Img = Image.open(image_name)
#     if Img.mode != 'RGB':
#         Img = Img.convert('RGB')
#     return Img
#
#
# def get_default_img_loader():
#     return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        # print('csv_file: ', csv_file)
        self.data = pd.read_table(csv_file, sep=' ', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        # self.data = self.data.iloc[0:1000]

        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        # self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            # print('csv_file: ',csv_file)
            a = self.data.iloc[index,0]
            # name = self.data.iloc[index, 0].split(' ')
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            # print('image_name:',image_name)
            I = Image.open(image_name)
            if self.transform is not None:
                I = self.transform(I)
            # mos = [i for i in name[1]]
            # print(name[1])
            # mos = []
            # for i in range(1):
            #     t = float(name[1])
            # #     mos.append(t)
            # #     print(t)
            # mos = t.split()
            # mos1 = name[1]
            # mos2 = float(name[1])
            mos = self.data.iloc[index, 1]
            # print('mos:',mos)
            # std = self.data.iloc[index, 2]
            sample = {'I': I, 'mos': mos}
        else:
            # name = self.data.iloc[index, 0].split(' ')
            SCI_image1 = os.path.join(self.img_dir,'splits2', self.data.iloc[index, 0])+'.bmp'
            # print('SCI_image1:',SCI_image1)
            # print('image_name1', image_name1[1])
            SCI_image2 = os.path.join(self.img_dir, 'splits2', self.data.iloc[index, 1])+'.bmp'

            # print('SCI_image2:', SCI_image2)
            # print(name[-1])
            # yb_SCI = list(map(int, self.data.iloc[index, 2]))
            yb_SCI = self.data.iloc[index, 2]
            # print('yb_SCI:', yb_SCI)

            NI_image1 = os.path.join(self.img_dir,'splits2',  self.data.iloc[index, 3])
            # print('NI_image1:',NI_image1)
            # print('image_name1', image_name1[1])
            NI_image2 = os.path.join(self.img_dir,'splits2',  self.data.iloc[index, 4])
            # print('NI_image2:', NI_image2)
            # print(name[-1])
            yb_NI = self.data.iloc[index, 5]
            # print('yb_NI:', yb_NI)

            SCI1 = Image.open(SCI_image1)
            SCI2 = Image.open(SCI_image2)
            NI1 = Image.open(NI_image1)
            NI2 = Image.open(NI_image2)
            if self.transform is not None:
                SCI1 = self.transform(SCI1)
                SCI2 = self.transform(SCI2)
                NI1 = self.transform(NI1)
                NI2 = self.transform(NI2)

            sample = {'SCI1': SCI1, 'SCI2': SCI2, 'yb_SCI': yb_SCI,'NI1': NI1, 'NI2': NI2, 'yb_NI': yb_NI}

        return sample

    def __len__(self):
        return len(self.data.index)
