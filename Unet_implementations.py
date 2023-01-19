import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp
import File_Collector as fc

# Device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train and test data , single column df with file name
df_train = fc.File_Collector().create_df()[0]
df_train = fc.File_Collector().create_df()[1]

# Calling U-Net Function
model = smp.Unet('mobilenet_v2', encoder_weights='imagenet',
                 classes=27, activation=None, encoder_depth=5,
                 decoder_channels=[256, 128, 64, 32, 16])

model.to(device)
