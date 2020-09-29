from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch
import numpy
from torch.utils.data import Dataset
import copy
import random
from scipy.special import comb
import os
import pandas as pd
import numpy as np
import matplotlib.image as image
from scipy.ndimage.interpolation import rotate
import os
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
class toumorDataset(Dataset):


    def __init__(self, dataset_dir, preprocess=True, transform=None,image_size =128):

        self.dataset_dir= dataset_dir
        self.preprocess = preprocess
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        dest_dir = "/content/new"
        patients = sorted(os.listdir(dest_dir))
        return len(patients)

    def __getitem__(self, idx):

        slices = torch.zeros((1, 1, self.image_size, self.image_size))
        if idx >= 155:
          label = 1
        else:
          label = 0
        patient_dir = os.path.join(self.dataset_dir, str(idx)+".jpg")
        slice_data = image.imread(patient_dir,".jpg")
        # print("1",slice_data)
        # print(slice_data.shape)
        # s= cv2.cvtColor(slice_data , cv2.COLOR_RGB2GRAY)
        # print("2",s)
        # print(s.shape)
        # normalize all images to [0-255]
        slice_data = slice_data.astype(np.uint8)
        # slice_data *= 255.0 / np.max(slice_data)
        # print("slice_data.shape1")
        if self.transform:
            slice_data = (self.transform(slice_data)).numpy()*255
        # print("slice_data.shape")
        x_train_tensor = torch.from_numpy(np.true_divide(slice_data,255)).unsqueeze(0)
        # print(x_train_tensor.shape)
        # print(x_train_tensor[0][0][100])
        # print(x_train_tensor[0][0][50])

        sample = [x_train_tensor,label]
        return sample
