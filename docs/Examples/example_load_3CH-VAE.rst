Example: 3CH VAE
================

This example shows how to load in the 3CH VAE

.. code:: ipython3

    # import time as t
    import os, sys
    from pathlib import Path
    
    project_root = Path.cwd().parents[1]
    os.chdir(project_root)   # now cwd is .../pvcracks
    
    from pvcracks.vae.VAE_model_3CH import Encoder, Decoder, VAE
    import requests
    import torch


Set device for torch
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #GPU or CPU
    print(f"Are we using the GPU: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


.. parsed-literal::

    Are we using the GPU: True


Load 3CH VAE model
------------------

.. code:: ipython3

    from io import BytesIO
    
    # Load from Datahub
    url = "https://datahub.duramat.org/dataset/919a555d-dd97-46ad-b77c-ae7e8894e6c4/resource/e83785e1-ba34-4212-b519-c6535b3e6804/download/model_3ch_233_weights.pth"
    #Link from the project folder: https://datahub.duramat.org/dataset/pvcracks-trained-vae-model
    
    #Download from url
    response = requests.get(url)
    if response.status_code == 200:
        model = VAE(latent_dim=50)  # Create an instance of your model
        model.load_state_dict(torch.load(BytesIO(response.content), weights_only=True))
        model.to(device)  # Move to the appropriate device
    else:
        print(f"Failed to download model. Status code: {response.status_code}")
    
    #Evaluate model
    model.eval()


.. parsed-literal::

    [32mLinear(in_features=50176, out_features=50, bias=True)[0m
    [34mLinear(in_features=50176, out_features=50, bias=True)[0m




.. parsed-literal::

    VAE(
      (encoder): Encoder(
        (conv): Sequential(
          (0): Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (1): ReLU()
          (2): Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (3): ReLU()
          (4): Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (5): ReLU()
          (6): Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (7): ReLU()
          (8): Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (9): ReLU()
          (10): Conv2d(512, 1024, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (11): ReLU()
        )
        (fc_mu): Linear(in_features=50176, out_features=50, bias=True)
        (fc_logvar): Linear(in_features=50176, out_features=50, bias=True)
      )
      (decoder): Decoder(
        (fc): Linear(in_features=50, out_features=50176, bias=True)
        (deconv): Sequential(
          (0): ConvTranspose2d(1024, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (1): ReLU()
          (2): ConvTranspose2d(512, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
          (3): ReLU()
          (4): ConvTranspose2d(256, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))
          (5): ReLU()
          (6): ConvTranspose2d(128, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))
          (7): ReLU()
          (8): ConvTranspose2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))
          (9): ReLU()
          (10): ConvTranspose2d(32, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))
          (11): Sigmoid()
        )
      )
    )


