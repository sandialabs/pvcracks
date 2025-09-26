# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:04:49 2023

@author: jlbraid, nrjost
"""
import time as t
import sys
t0 = t.time() #timerimport numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset, FixedRotation, FixedHorizontalFlip, FixedVerticalFlip
from termcolor import colored
from pytorch_ssim import SSIM #don't use pip installed version, is not maintained
from torchvision import transforms
import os
from pathlib import Path

from VAE_functions import preprocess, vae_loss, encode_image, decode_latent_vector, show_input_output_images, ssim_input_output, generate_random_images, show_generated_images, set_seeds

#Set deterministic backend
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Set seed
# set_seeds(30, True)

#emmas pv-vision crack masks from new images, full dataset available below
#https://datahub.duramat.org/dataset/pvcracks-crack-masks-for-vae
new_pathimages = '/home/nrjost/DuramatCrackImg/emma_pvvision_crack_images/'
root = Path(new_pathimages)
files = os.listdir(root)
files_img = [x for x in files if x.endswith(('tiff'))]
array_new=[]
for i in range(len(files_img)):
    dat=io.imread(f"{root}/{files_img[i]}")
    datmean = dat.mean()
    dat = (dat > datmean).astype(np.float32)
    array_new.append(dat)
array_new = np.stack(array_new)
print("Size of new array with cracked images %s" % str(array_new.shape))

#add new images to training array
# train =  np.concatenate((train, array_new[:,:,:,1:2]), axis = 0)
train = array_new[:-100,:,:,1:2]
test = array_new[-99:,:,:,1:2]

print("Size of NEW training array with crack %s" % str(train.shape))

# Hyperparameters
latent_dim = 50
batch_size = 16
learning_rate = 1e-3
num_epochs = 250

#Loss weigths
bce_weight = 0.01 #0.01
ssim_weight = 10000 #10000
kld_weight = 0.3 #0.3

transform = transforms.Compose([FixedRotation(angle=180), 
                                FixedHorizontalFlip(),
                                FixedVerticalFlip()])

# Create a CustomDataset instance
train = CustomDataset(train)
test = CustomDataset(test)
train_augmented = CustomDataset(train, transform=transform)
print(f"Augmented training set has {len(train_augmented)} images")
test_augmented = CustomDataset(test, transform=transform) 

# Create a DataLoader instance
# train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=lambda worker_id: set_seed(42 + worker_id))
# train_loader = DataLoader(train_augmented, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=lambda worker_id: set_seed(42 + worker_id))
train_loader = DataLoader(train_augmented, batch_size=batch_size, shuffle=True, num_workers=1)

#Load Encoder, Decoder, VAE
from VAE_model import Encoder, Decoder, VAE

# Initialize the VAE model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
model = VAE(latent_dim)
model.to(device)
#model = VAE(latent_dim).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
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
        loss = vae_loss(recon_x=recon_data, x = data, mu = mu, logvar = logvar, bce_weight = bce_weight, kld_weight = kld_weight, ssim_weight = ssim_weight)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()/len(data):.6f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train):.6f}')
    epoch_loss = train_loss / len(train)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    if len(train_losses) > 35:
        if epoch_loss >= train_losses[-25]:
            if epoch_loss >= train_losses[-30]:
                if epoch_loss >= train_losses[-35]:
                    sys.exit('Training loss stuck, Overfitting. Current loss %f > loss 25, 30 & 35 epochs ago %f' % (epoch_loss, train_losses[-25]))


print(mu.size())
print(logvar.size())
print(colored(train_losses, 'blue'))
print(colored(num_epochs, 'yellow'))

# Plot the training loss per epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses)
# plt.ylim(10000, 1000)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.savefig('CurrentResults/Trainingloss.png')
plt.show()

import matplotlib.pyplot as plt

num_images=99
test_loader = DataLoader(test,batch_size=num_images,shuffle=True,num_workers=4)
# test_loader = DataLoader(test_augmented,batch_size=num_images,shuffle=True,num_workers=4)

# Load a batch of images from the train_loader
images = next(iter(test_loader))
images = images.to(device)

# Get the VAE outputs for the input images
with torch.no_grad():
    vae_outputs, _, _ = model(images)
    
        
# Display 5 input images and their VAE outputs
num_images_to_display = 5
show_input_output_images(images, vae_outputs, num_images_to_display, path='CurrentResults/')
num_images_to_compare = 99
ssim_comp = ssim_input_output(images, vae_outputs, num_images_to_compare, path='CurrentResults/')

# Generate new images using the VAE model
num_images = 10  # Number of images to generate
generated_images, _ = generate_random_images(model, num_images, latent_dim)
generated_images.to("cpu")

print(f"Shape of Generate images = {generated_images.shape}")

show_generated_images(generated_images, num_images, path='CurrentResults/')

#Save the model for reimporting
# torch.save(model.state_dict(), 'model.pth')
model.to('cpu')
torch.save(model, 'model_newimg_upepochs.pth')

#time full code
t1 = t.time()
total = t1-t0
print('Time running code %s minutes' % (str(total/60)))