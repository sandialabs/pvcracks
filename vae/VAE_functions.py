# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:04:49 2023

@authors: jlbraid, nrjost
"""

import sys
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from termcolor import colored
from pytorch_ssim import SSIM #don't use pip installed version (not maintained). Use: https://github.com/Po-Hsun-Su/pytorch-ssim
from torchvision import transforms
import os

def set_seeds(seed=50, multiGPU=False):
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        The seed value to use for random number generators.
    multiGPU : bool
        If True, sets seeds for multiple GPUs.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function sets seeds for Python's random module, numpy, and PyTorch (CPU and CUDA).
    """
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if multiGPU:
            torch.cuda.manual_seed_all(seed)
    #test
    torch_rand = []
    np_rand = []
    py_rand = []
    for i in range(0, 100):
        torch_rand.append(torch.rand(1))
        np_rand.append(np.random.rand(1))
        py_rand.append(random.random())
    print(f"Mean of torch random = {sum(torch_rand)/len(torch_rand)}")
    print(f"Mean of numpy random = {sum(np_rand)/len(np_rand)}")
    print(f"mean of random random = {sum(py_rand)/len(py_rand)}")

    

def preprocess(impath):
    """
    Preprocess an image from the specified path.
    
    Parameters
    ----------
    impath : str
        Path to the image file.
    
    Returns
    -------
    numpy.ndarray
        The preprocessed image as a numpy array.
    
    Notes
    -----
    This function reads an image, normalizes its pixel values to the range [0, 1], and converts it to a float32 numpy array.
    """
    #Preprocess data as a float 0. to 1.
    dat=io.imread(impath)
    dat=dat[:,:,:2]/255
    return dat.astype('float32')

def vae_loss(recon_x, x, mu, logvar, bce_weight, kld_weight, ssim_weight, device='cuda'):
    """
    Compute the loss for the Variational Autoencoder (VAE).
    
    Parameters
    ----------
    recon_x : torch.Tensor
        The reconstructed images.
    x : torch.Tensor
        The original input images.
    mu : torch.Tensor
        The mean of the latent distribution.
    logvar : torch.Tensor
        The log variance of the latent distribution.
    bce_weight : float
        Weight for the Binary Cross Entropy (BCE) loss.
    kld_weight : float
        Weight for the Kullback-Leibler Divergence (KLD) loss.
    ssim_weight : float
        Weight for the Structural Similarity Index (SSIM) loss.
    device : str
        The device to perform computations on.
    
    Returns
    -------
    torch.Tensor
        The total loss.
    
    Notes
    -----
    The function combines BCE loss, KLD loss, and SSIM loss to compute the total VAE loss.
    Uses: from pytorch_ssim import SSIM 
    don't use pip installed version (not maintained). Use: https://github.com/Po-Hsun-Su/pytorch-ssim

    """ 
    #minimizing the elbow, evidence based lower bound
    print(colored("Shape of x is", 'magenta'))
    print(colored(x.shape, 'magenta'))
    print(colored(("Shape of recon_x is"), 'cyan'))
    print(colored(recon_x.shape, 'cyan'))
    recon_loss = nn.functional.binary_cross_entropy(recon_x.view(-1, 400*400), x.view(-1, 400*400), reduction='sum') #adapt to size of input array
    ssim_loss = SSIM(window_size=50) #18, 50
    ssimloss = 1 - ssim_loss(recon_x, x)
    ssimloss = ssimloss.to(device)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss.to(device)
    print("Current BCE loss =%f" % recon_loss)
    print("Current SSIM loss =%f" % ssimloss)
    print("Current KLD loss =%f" % kld_loss)
    total_loss = (
        bce_weight * recon_loss +
        kld_weight * kld_loss +
        ssim_weight * ssimloss
    )
    return total_loss

def initialize_model_optimizer(model, latent_dim, learning_rate, device):
    """
    Initialize the VAE model and optimizer.
    
    Parameters
    ----------
    model : torch.nn.Module
        The VAE model class.
    latent_dim : int
        The dimension of the latent space.
    learning_rate : float
        The learning rate for the optimizer.
    device : str = "cpu" or "cuda"
        The device to perform computations on.
    
    Returns
    -------
    tuple
        The initialized model and optimizer.
    
    Notes
    -----
    The function initializes a VAE model and an Adam optimizer.
    """
    device = torch.device(device)
    model = VAE(latent_dim)
    model.to(device)
    #model = VAE(latent_dim).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model):
    """
    Train the VAE model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The VAE model to be trained.
    
    Returns
    -------
    tuple
        The mean, log variance, training losses, and number of epochs.
    
    Notes
    -----
    The function trains the VAE model for a specified number of epochs and returns the training losses.
    """
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # print(f"Shape of data in training loop: {data.shape}")
            data = data.to(device)
            # data = data.to("cuda")
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            recon_data = recon_data.to(device)
            loss = vae_loss(recon_data, data, mu, logvar, bce_weight, kld_weight, ssim_weight)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()/len(data):.6f}')
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train):.6f}')
        epoch_loss = train_loss / len(train)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
        if len(train_losses) > 33:
            if epoch_loss >= train_losses[-25]:
                sys.exit('Training loss stuck, Overfitting. Current loss %f, loss 25 epochs ago %f' % (epoch_loss, train_losses[-30]))
    return mu, logvar, train_losses, num_epochs
            

def plot_training_losses(num_epochs, train_losses, path):
    """
    Plot the training losses per epoch.
    
    Parameters
    ----------
    num_epochs : int
        The number of training epochs.
    train_losses : list
        A list of training losses.
    path : str or None
        Path to save the plot.
    
    Returns
    -------
    None
    
    Notes
    -----
    """
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses)
    # plt.ylim(10000, 1000)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Epoch')
    if path:
        plt.savefig(''+path+'/Trainingloss.png')
    plt.show()

def encode_image(model, image):
    """
    Encode an image using the VAE encoder.
    
    Parameters
    ----------
    model : torch.nn.Module
        The VAE model.
    image : torch.Tensor
        The input image tensor.
    
    Returns
    -------
    torch.Tensor
        The latent vector.
    
    Notes
    -----
    """
    import torch
    model.eval()

    image = image.unsqueeze(0)

    mu, logvar = model.encoder(image)

    latent_vector = model.reparameterize(mu, logvar)

    return latent_vector


def decode_latent_vector(model, latent_vector):
    """
    Decode a latent vector using the VAE decoder.
    
    Parameters
    ----------
    model : torch.nn.Module
        The VAE model.
    latent_vector : torch.Tensor
        The latent vector.
    
    Returns
    -------
    torch.Tensor
        The reconstructed image.
    
    Notes
    -----
    """
    model.eval()

    reconstructed_image = model.decoder(latent_vector)

    return reconstructed_image

def load_from_testloader(test, num_images=100):
    """
    Load a batch of images from the test dataset.
    
    Parameters
    ----------
    test : torch.utils.data.Dataset
        The test dataset.
    num_images : int
        The number of images to load in a batch.
    
    Returns
    -------
    torch.Tensor
        A batch of images.
    
    Notes
    -----
    """
    test_loader = DataLoader(test,batch_size=num_images,shuffle=True,num_workers=4)
    # test_loader = DataLoader(test_augmented,batch_size=num_images,shuffle=True,num_workers=4)
    images = next(iter(test_loader))
    images = images.to(device)
    
def VAE_output_for_images(model, images):
    """
    Get the VAE outputs for the input images.
    
    Parameters
    ----------
    model : torch.nn.Module
        The VAE model.
    images : torch.Tensor
        The input images.
    
    Returns
    -------
    torch.Tensor
        The VAE outputs.
    
    Notes
    -----
    """

    with torch.no_grad():
        vae_outputs, _, _ = model(images)
    return vae_outputs
    
def show_input_output_images(inputs, outputs, num_images, path):
    """
    Display the input images and their VAE outputs.
    
    Parameters
    ----------
    inputs : torch.Tensor
        The input images.
    outputs : torch.Tensor
        The VAE output images.
    num_images : int
        The number of images to display.
    path : str
        Path to save the plot.
    
    Returns
    -------
    None
    
    Notes
    -----
    The function displays the input images and their VAE outputs side by side.
    """
    from skimage.metrics import structural_similarity
    num_cols = 2  # One column for input images and one column for VAE outputs
    num_rows = num_images
    #copy images to CPU for plotting
    inputs = inputs.cpu()
    outputs = outputs.cpu()

    plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    for i in range(num_images):
        # Display input image
        plt.subplot(num_rows, num_cols, 2 * i + 1)
        plt.imshow(inputs[i].squeeze(0).numpy(), cmap='gray')
        #plt.title("Input")
        plt.axis('off')

        # Display VAE output
        (score, diff) = structural_similarity(inputs[i].squeeze(0).numpy(), outputs[i].squeeze(0).numpy(), full=True)
        plt.subplot(num_rows, num_cols, 2 * i + 2)
        plt.imshow(outputs[i].squeeze(0).numpy(), cmap='gray')
        #plt.title("VAE Output")
        plt.axis('off')
        plt.text(0, 0, 'SSIM comparison '+str(round(score,4)*100)+'%')
        if path:
            plt.savefig(''+path+'/exampl_input_output.png')

    plt.show()

def ssim_input_output(inputs, outputs, num_images, path):
    """
    Compute the SSIM between input and output images and save the results.
    
    Parameters
    ----------
    inputs : torch.Tensor
        The input images.
    outputs : torch.Tensor
        The VAE output images.
    num_images : int
        The number of images to compare.
    path : str or None
        Path to save the results.
    
    Returns
    -------
    pandas.DataFrame
        The SSIM comparison results.
    
    Notes
    -----
    The function computes the SSIM between input and output images and saves the results to a CSV file.
    """
    from skimage.metrics import structural_similarity
    inputs = inputs.cpu()
    outputs = outputs.cpu()
    ssim_comp = pd.DataFrame()
    num_rows = num_images
    for i in range(num_images):
        (score, diff) = structural_similarity(inputs[i].squeeze(0).numpy(), outputs[i].squeeze(0).numpy(), full=True)
        listssim = pd.DataFrame(data=[[i, score, diff]], columns=['num', 'ssim_score', 'ssim_array'])
        ssim_comp = pd.concat([ssim_comp, listssim], axis=0)
    if path:
        ssim_comp.to_csv(''+path+'/ssim_comp.csv')
    plt.figure()
    ssim_comp['ssim_score'].plot(kind='box', title='SSIM value of all images')
    plt.axhline(y=0.95, color='r', linestyle='--')
    if path:
        plt.savefig(''+path+'/SSIM_compare.png')
    return ssim_comp
        
def generate_random_images(model, num_images, latent_dim, device="cuda"):    
    """
    Generate random images using the VAE model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The VAE model.
    num_images : int
        The number of images to generate.
    latent_dim : int
        The dimension of the latent space.
    device : str = "cpu" or "cuda"
        The device to perform computations on.
    
    Returns
    -------
    tuple
        The generated images and the random latent vectors.
    
    Notes
    -----
    """
    model.eval()
    random_latent_vectors = torch.randn(num_images, latent_dim).to(device)
    generated_images = model.decoder(random_latent_vectors).to(device)
    return generated_images, random_latent_vectors

def show_generated_images(generated_images, num_images, path):
    """
    Display the generated images.
    
    Parameters
    ----------
    generated_images : torch.Tensor
        The generated images.
    num_images : int
        The number of images to display.
    path : str or None
        Path to save the plot.
    
    Returns
    -------
    None
    
    Notes
    -----
    """
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    generated_images = generated_images.to("cpu")
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))

    for i, img in enumerate(generated_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img.squeeze(0).detach().numpy(), cmap='gray')
        plt.axis('off')
    if path:
        plt.savefig(''+path+'/GenImages.png')
    plt.show()
