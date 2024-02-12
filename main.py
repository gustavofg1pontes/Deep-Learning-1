# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from Modelo import Modelo, treino, validacao

transform = transforms.ToTensor()

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modelo = Modelo()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    treino(modelo, trainloader, device)
    validacao(modelo, valloader, device)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
