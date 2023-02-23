import os
from glob import glob

import SimpleITK as sitk
import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F

from model.RegistrationNetworks import euler_angles_to_matrix


class Dataset(torch.utils.data.Dataset):
    """
    GENERAL DATASET
    """
    def __init__(self, train_val_test):
        super().__init__(train_val_test)
        self.overfit = False
        self.train_val_test = train_val_test

    @staticmethod
    def get_image_header(path):
        image = sitk.ReadImage(path)
        dim = np.array(image.GetSize())
        voxel_sp = np.array(image.GetSpacing())
        return dim[::-1], voxel_sp[::-1]

    def adjust_shape(self, multiple_of=16):
        old_shape, _ = self.get_image_header(self.moving_img[0])
        new_shape = tuple([int(np.ceil(shp / multiple_of) * multiple_of) for shp in old_shape])
        self.inshape = new_shape
        self.offsets = [shp - old_shp for (shp, old_shp) in zip(new_shape, old_shape)]

    def read_image_sitk(self, path):
        if os.path.exists(path):
            image = sitk.ReadImage(path)
        else:
            print('image does not exist')
        return image

    def read_image_np(self, path):
        if os.path.exists(path):
            image = sitk.ReadImage(path)
            image_np = sitk.GetArrayFromImage(image).astype('float32')
        else:
            print('image does not exist')
        return image_np

    def transform_rigid(self, image_t):
        rand_trans = np.random.uniform(low=-self.max_trans, high=self.max_trans, size=(3,)).astype('float32')
        rand_angles = np.random.uniform(low=-self.max_angle, high=self.max_angle, size=(3,)).astype('float32')

        if self.overfit:
            rand_trans = np.array([0, 0.1, -0.05]).astype('float32')
            rand_angles = np.array([-10, 4, 12]).astype('float32')

        translation = torch.from_numpy(rand_trans).unsqueeze(0)
        euler_angles = np.pi * torch.from_numpy(rand_angles).unsqueeze(0) / 180.

        # rotation
        rot_mat = euler_angles_to_matrix(euler_angles=euler_angles, convention="XYZ")

        # get ground truth transformation
        T = torch.cat([rot_mat.squeeze(), translation.squeeze().view(3, 1)], axis=1)
        T = T.view(-1, 3, 4)

        # do rigid augmentation
        grid = F.affine_grid(T, image_t.unsqueeze(0).size())
        image_aug_t = F.grid_sample(image_t.unsqueeze(0), grid).squeeze(0)
        return image_aug_t, T

    def overfit_one(self, i):
        self.overfit = True
        self.moving_img = [self.moving_img[i]]

    def __len__(self):
        return len(self.moving_img)

    def __getitem__(self, i):
        """Load the image/label into a tensor"""
        moving_np = self.read_image_np(self.moving_img[i])
        moving_t = torch.from_numpy(moving_np).unsqueeze(0)
        fixed_t, T = self.transform_rigid(moving_t)
        return moving_t, fixed_t, T

class DatasetLung(Dataset):
    def __init__(self, train_val_test, version):
        self.set = 'lung'
        self.extension = '.nii.gz'
        self.version = version
        self.img_folder = f'/home/ikolenbrander/Documents/PhD_IMAG/999_DATA/LUNG_4DCT/DATA/PREPROCESSED/PREPROCESSED_V2.1/splitA/{train_val_test}/image/***'
        self.init_paths()
        self.inshape, self.voxel_spacing = self.get_image_header(self.moving_img[0])
        self.adjust_shape(multiple_of=32)
        self.max_trans = 0.2
        self.max_angle = 30

    def init_paths(self):
        self.phases_moving = range(0, 90, 10)

        # Get all file names inside the data folder
        self.img_paths= glob(self.img_folder)
        self.img_paths.sort()
        self.moving_img = []

        for img_folder in self.img_paths:
            for phase_moving in self.phases_moving:
                m = os.path.join(img_folder, 'T{:02d}{}'.format(phase_moving, self.extension))
                if os.path.exists(m):
                    self.moving_img.append(m)