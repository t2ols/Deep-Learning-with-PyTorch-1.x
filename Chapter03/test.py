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


# if __name__ == '__main__':
#     freeze_support()


#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
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
    # plt.show()

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

#torch load ##########################

simple_transform = transforms.Compose([transforms.Resize((224,224))
                                    ,transforms.ToTensor()
                                    ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train = ImageFolder(os.path.join(path,'train'),simple_transform)
valid = ImageFolder(os.path.join(path,'valid'),simple_transform)

# print(train.class_to_idx)
# print(train.classes) 

# imshow(train[50][0])

######################################

#Dataloader with worker
#window num_workers = 0
train_data_gen = torch.utils.data.DataLoader(train, shuffle=True,batch_size=64, num_workers=0)
valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=0)


dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
dataloaders = {'train':train_data_gen,'valid':valid_data_gen}

model_ft = models.resnet18(pretrained=True)  #1000여개 분류코드
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)   #개, 고양이 분류 2 개

# print( model_ft )

if is_cuda :
    model_ft.cuda()

#leanring
learning_rate = 0.001
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # running_loss += loss.data[0]
                running_loss += loss.data #single
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Loss and Optimizer
criterion = nn.CrossEntropyLoss() #loss
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  #optimizer

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) #학습률을 변경


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)    