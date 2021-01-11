import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
# from torchvision import transforms
# from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
# from torchvision.datasets import ImageFolder
# from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore")
import time

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)   

path = 'Chapter03/'    

dog_files = [f for f in glob.glob(path + 'Dog-Cat-Classifier/Data/Train_Data/dog/*.jpg')]
cat_files = [f for f in glob.glob(path + 'Dog-Cat-Classifier/Data/Train_Data/cat/*.jpg')]
files = dog_files + cat_files
print(f'Total no of images {len(files)}')