""" THIS IS A SCRIPT TO TRAIN A SWIN TRANSFORMER FOR RIGID REGISTRATION """
from datasets import DatasetLung
from model.RegistrationNetworks import RegTransformer
from model.configurations import get_VitBase_config
from train_val_test import train_epoch
from utils import set_seed
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    random_seed = 10
    overfit = True  # Set this one to True if you want to overfit on only one image.
    learning_rate = 0.00000001  # Tune this hyperparameter
    batch_size = 1
    epochs = 300
    device = torch.device('cuda')

    set_seed(random_seed)
    train_dataset = DatasetLung('train', version='2.1A')
    if overfit:
        train_dataset.overfit_one(i=0)

    moving, fixed, _ = train_dataset[0]

    """ CONFIG NETWORK """
    config = get_VitBase_config(img_size=tuple(train_dataset.inshape))
    model = RegTransformer(config)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    """ TRAINING """
    print('\n----- Training -----')
    loss_list = list()
    T_error_list = list()
    epoch = 1
    while epoch < epochs:
        print(f'\n[epoch {epoch} / {epochs}]')
        loss, img_warped, T_error = train_epoch(model, train_loader, optimizer, device)
        loss_list.append(loss)
        T_error_list.append(T_error)
        epoch += 1

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(moving.squeeze().cpu().numpy()[:, 80, :], cmap='gray')
    ax[0].set_title('moving image')
    ax[1].imshow(fixed.squeeze().cpu().numpy()[:, 80, :], cmap='gray')
    ax[1].set_title('fixed image')
    ax[2].imshow(img_warped.squeeze().detach().cpu().numpy()[:, 80, :], cmap='gray')
    ax[2].set_title('warped image')
    fig.suptitle('Coronal image slices')
    fig.show()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(loss_list)
    ax[0].set_title('loss function')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss (NCC)')
    ax[1].plot(T_error_list)
    ax[1].set_title('T mean squared error (ground truth - predicted)')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('MSE')
    fig.show()