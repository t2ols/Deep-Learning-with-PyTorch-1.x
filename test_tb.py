import torch
import torchvision
import os

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

from multiprocessing import Process, freeze_support

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    freeze_support()

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = torchvision.models.resnet50(False)

    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)

    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()