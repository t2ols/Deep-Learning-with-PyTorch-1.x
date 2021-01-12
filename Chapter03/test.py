import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore")
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#torch install cpu or gpu
#window install : https://lsjsj92.tistory.com/494
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

no_of_images = len(files)

#셔플된 이미지 인덱스
shuffle = np.random.permutation(no_of_images)

if not os.path.isdir(os.path.join(path,'train')) :
    os.mkdir(os.path.join(path,'train'))

if not os.path.isdir(os.path.join(path,'valid')) :
    os.mkdir(os.path.join(path,'valid'))

for t in ['train', 'valid']:
    for folder in ['dog/', 'cat/']:
        if not os.path.isdir(os.path.join(path,t,folder)) : 
            os.mkdir(os.path.join(path,t,folder))
        else :
            shutil.rmtree(os.path.join(path,t,folder)) #디렉토리 초기화
            os.mkdir(os.path.join(path,t,folder))

for i in shuffle[:250]:
    folder = files[i].split('\\')[-1].split('.')[0]
    image = files[i].split('\\')[-1]
    #os.rename(files[i],os.path.join(path,'valid',folder,image))
    shutil.copy(files[i], os.path.join(path,'valid',folder,image))


for i in shuffle[251:501]:
    folder = files[i].split('\\')[-1].split('.')[0]
    image = files[i].split('\\')[-1]
    # os.rename(files[i],os.path.join(path,'train',folder,image))
    shutil.copy(files[i], os.path.join(path,'train',folder,image))

is_cuda = False
if torch.cuda.is_available():
    print('CUDA available')
    is_cuda = True

simple_transform = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train = ImageFolder(os.path.join(path,'train'),simple_transform)
valid = ImageFolder(os.path.join(path,'valid'),simple_transform)

print(train.class_to_idx)
print(train.classes) 

print(train.classes) 


imshow(train[50][0])