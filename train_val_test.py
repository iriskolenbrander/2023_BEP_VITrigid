"""
Training/validation functions
"""
import sys

import torch
from tqdm import tqdm
from utils import NCC


def train_epoch(model, data_loader, optimizer, device):
    """
    Train for one epoch
    """
    # Creates a GradScaler once at the beginning of training.
    total_loss = 0

    # Initialize loss functions
    similarity_loss = NCC(device)
    mse_loss = torch.nn.MSELoss()
    model.train()
    for batch_idx, (img_moving, img_fixed, T_ground_truth) in enumerate(tqdm(data_loader, file=sys.stdout)):
        # Take the img_moving and fixed images to the GPU
        img_moving, img_fixed = img_moving.to(device), img_fixed.to(device)

        # Zero out the optimizer gradients as these are normally accumulated
        optimizer.zero_grad()

        """ Loss calculation """
        # Get the transformed img_moving image and the corresponding Displacement Vector Field
        img_warped, T = model(img_moving, img_fixed)

        # Compute the similarity loss between the transformed img_moving image and the fixed image (ignoring padding) and back propagate
        loss = similarity_loss.forward(img_fixed, img_warped)
        loss.backward()
        optimizer.step()

        # Update the total loss
        total_loss += loss.item()
        T_error = mse_loss(T, T_ground_truth.squeeze(0).to(device)).item()

    """ Print loss """
    print("Loss = %.2f" % (total_loss / len(data_loader)))
    return total_loss / len(data_loader), img_warped, T_error

