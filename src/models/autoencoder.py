###############################################################################
# The purpose of this module is to design an autoencoder which learns a
# meaningful compressed representation of the 4 structural MRI scans and
# compresses them down into three feature maps without changing the
# spatial dimensions.
###############################################################################
import os
import numpy as np
import torch
import torch.nn as nn
from livelossplot import PlotLosses
from tqdm.auto import tqdm


class EarlyStopper:
    """
    Creates an EarlyStopper object which can be used to
    cease training after a defined number of epochs without
    training loss reduction

    based on https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=1, min_delta=0.):
        """
        Creates an EarlyStopper object which can be used to
        cease training after a defined number of epochs without
        traning loss reduction

        Inputs:
            patience - int, default 1, number of epochs to stop training
            without loss reduction

            min_delta - float, default 0., threshold value for training loss
        """
        self.patience = patience
        self.min_delta = min_delta
        # tracks how many iterations have passed without improvement
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss):
        """
        helper function that compares current loss to global
        minimum loss, and updates patience counter if current
        loss is not less than global minimum loss. If the
        patience counter threshold is exceeded, training is stopped

        Inputs:
            loss - float, current epoch training loss

        Returns a boolean
        """
        if loss < self.min_loss:
            # update min_loss
            self.min_loss = loss
            # reset the counter
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            # the current loss is greater than the minimum
            # plus a threshold criteria
            # increment patience counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Autoencoder(nn.Module):
    def __init__(self, input_nchannels=4, version=None):
        """
        Defines an autoencoder object that learns a compressed
        representation of the 4 MRI structural scans down to
        3 feature maps, without changing spatial dimensions

        inputs:
            input_nchannels - integer, default is 4, expects 4
            version - string, default is None for baseline, 'v1' for other
        """
        super(Autoencoder, self).__init__()
        assert input_nchannels == 4, "Expected number of input channels is 4"
        self.input_nch = 4
        self.version = version

        # check whether to use baseline or different version
        if not self.version:
            # implement baseline model
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
        elif self.version == 'v1':
            # implement version 1 that is the same as baseline
            # but does not include sigmoid activation in decoder output
            # encoder
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

            # decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=3,
                    out_channels=self.input_nch,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        elif self.version == 'v2':
            # v2 encoder 4 --> 12 --> 24 -- > 3
            # decoder 3 --> 24 --> 12 --> 4
            # encoder
            self.encoder = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.input_nch,
                    out_channels=12,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=12,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=24,
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
            )
            # decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=3,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    in_channels=24,
                    out_channels=12,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    in_channels=12,
                    out_channels=self.input_nch,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        elif self.version == 'v3':
            # v3 encoder 4 --> 12 --> 24 --> 48 --> 3
            # decoder 3 --> 48 --> 24 --> 12 --> 4
            # encoder
            self.encoder = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.input_nch,
                    out_channels=12,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=12,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=24,
                    out_channels=72,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=72,
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                    padding="same"
                ),
                nn.ReLU(),
            )
            # decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=3,
                    out_channels=72,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    in_channels=72,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    in_channels=24,
                    out_channels=12,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.ConvTranspose3d(
                    in_channels=12,
                    out_channels=self.input_nch,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
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


def autoencoder_training_loop(model, loss_fn, optimizer, dataloader, nepochs=100, early_stopping=False, patience=10, min_delta=1e-4, name='model', checkpoint=True, chkpt_path='/content/models/checkpoints/', best_path='/content/models/best/'):
    """
    Implements a custom training loop for the autoencoder

    Inputs:
        model - an instance of the Autoencoder class
        loss_fn - an instance of a PyTorch loss function class
        optimizer - an instance of a PyTorch optimizer class
        dataloader - an instance of a Dataloader class for AutoencoderMRIDataset class
        nepochs - number of epochs for training, default 100
        early_stopping - boolean, default False
        patience - int, default 10, number of epochs without training loss reduction to stop training
        min_delta - float, default 1e-4, threshold tolerance training loss in early stopping
        name - string name for saving model, ex: 'baseline_model'
        checkpoint - boolean indicating whether to checkpoint save
        chkpt_path - string for saving model checkpoints
        best_path - string for saving current best model

    Returns a fitted model
    """
    # check whether save paths exist, create them if not
    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    # instantiate a livelossplot PlotLosses class
    liveloss = PlotLosses()
    loss_train = []

    # specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # send the model and loss function to the device
    model.to(device)
    loss_fn.to(device)

    # set a min loss value for checkpoint saving
    min_loss = 1e9

    # instantiate an EarlyStopper object, if specified
    if early_stopping:
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    # enter the training loop
    for epoch in range(1, nepochs + 1):

        print(f"Starting epoch: {epoch}")

        # create an empty dictionary for the loss logs
        logs = {}

        # track running loss and batch size
        running_loss = 0.0
        num_samples = 0

        # set model to train
        model.train()

        for batch in tqdm(dataloader):
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

            # capture current running loss and batch size
            batch_size = batch["vol"].size(0)
            running_loss += loss.detach() * batch_size
            num_samples += batch_size

        # record loss at end of epoch
        epoch_loss = running_loss / num_samples
        logs['loss'] = epoch_loss.item()
        loss_train.append(epoch_loss)

        # update the loss plot
        liveloss.update(logs)
        liveloss.send()

        # check-point save every 10 epochs
        if checkpoint and epoch % 10 == 0:
            outname = f"{name}_checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(chkpt_path, outname))

        # update min_loss and save if current best after nepochs // 2
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            if epoch > nepochs // 4:
                outname = f"{name}_current_best_epoch_{epoch}.pt"
                torch.save(model.state_dict(), os.path.join(best_path, outname))

        # check whether to stop training early
        if early_stopping:
            if early_stopper.early_stop(epoch_loss):
                print(f"Stopping early at epoch {epoch}")
                break


def normalize_channels(mri_tensor):
    """
    Normalize channels to range from 0 to 1

    Inputs:
        mri_tensor - a 4D torch tensor object in CHWZ format

    Returns a tensor object
    """
    num_channels = mri_tensor.size()[0]
    normalized_mri_tensor = mri_tensor.clone()

    for channel in range(num_channels):
        chan_min = mri_tensor[channel, :, :, :].min()
        chan_max = mri_tensor[channel, :, :, :].max()
        normalized_mri_tensor[channel, :, :, :] = ((normalized_mri_tensor[channel, :, :, :] - chan_min) /
                                                   (chan_max - chan_min))

    return normalized_mri_tensor
