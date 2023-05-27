###############################################################################
# The purpose of this module is to design an autoencoder which learns a
# meaningful compressed representation of the 4 structural MRI scans and 
# compresses them down into three feature maps without changing the 
# spatial dimensions.
###############################################################################
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_nchannels=4):
        """
        Defines an autoencoder object that learns a compressed
        representation of the 4 MRI structural scans down to 
        3 feature maps, without changing spatial dimensions
        """
        super(Autoencoder, self).__init__()
        assert input_nchannels == 4, "Expected number of input channels is 4"
        self.input_nch = 4
        
        # implement encoder
        # Input 4 channels --> Latent Space 3 channels
        # with same spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.input_nch,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding="same"
            ),
            nn.ReLU()
        )

        # implement decoder
        # Latent Space 3 channels --> Output 4 channels
        # with same spatial dimensions as input
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=3,
                out_channels=self.input_nch,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Sigmoid()
        )

        # initialize parameters
        self.reset_parameters()


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                # apply He normal initialization for weights and zero bias
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.ConvTranspose3d):
                # apply He normal initialization for weights and zero bias
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def autoencoder_training_loop(model, loss_fn, optimizer, dataloader, nepochs=100):
    """
    Implements a custom training loop for the autoencoder

    Inputs:
        model - an instance of the Autoencoder class
        loss_fn - an instance of a PyTorch loss function class
        optimizer - an instance of a PyTorch optimizer class
        dataloader - an instance of a Dataloader class for AutoencoderMRIDataset class
        nepochs - number of epochs for training, default 100

    Returns a fitted model
    """
    # specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # send the model and loss function to the device
    model.to(device)
    loss_fn.to(device)

    # enter the training loop
    for epoch in range(nepochs):
        for batch in dataloader:
            # move batch to device
            batch_current = batch["vol"].to(device, dtype=torch.float)

            # zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_current)

            # compute the loss
            loss = loss_fn(outputs, batch_current)

            # backward pass and optimization
            loss.backward()
            optimizer.step()
        
        # print the status
        print(f"Epoch {epoch + 1}/{nepochs} - Loss: {loss.item()}")