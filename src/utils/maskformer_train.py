'''Functions for maskformer training'''
import torch

from livelossplot import PlotLosses
from tqdm.auto import tqdm
from transformers import MaskFormerModel
from utils.data_handler import DataHandler

def peek(dataset, index):
    """
        Helper to print info about dataset at a specified index

        Args:
            index (int): specific index in the dataset
            dataset (MaskformerMRIDataset): train, test, or val MaskformerMRIDataset dataset
        Returns:
            A dictionary containing information about the dataset
    """
    input = dataset[index]
    for k,v in input.items():
        if isinstance(v, torch.Tensor):
            print(k,v.shape)

    print("dataset length", len(dataset))
    print("class labels", input["class_labels"])
    return input

class MaskFormerTrain():
    """
    MaskFormerTrain class defines an object for training using a trained MaskFormer model.
    
    The purpose of having a class is to allow testing the same model and parameters for datasets
    """
    
    def __init__(self, model: MaskFormerModel, n_epoch: int):
        
        # training default settings
        self.n_epoch = n_epoch
        self.save_interval = 5
        self.initial_lr = 1e-4
        self.model = model
        
        # file settings
        self.data_handler = DataHandler()
        
    def train(self, train_dataloader, val_dataloader, train_id, train_dir_prefix, optimizer=None, scheduler=None):
                        
        liveloss = PlotLosses()

        batch_max = 100000
        loss_train = []
        loss_val = []
        min_loss = 1e9
        
        # mutiplier is used to scale loss values to be more readable
        loss_multiplier = 10.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        
        # store initial learning rate from provided optimizer
        optimizer_state = optimizer.state_dict()
        scheduler_state = {}
        if scheduler is not None:
            scheduler_state = scheduler.state_dict()
        
        logs = {}
        for epoch in range(1,self.n_epoch+1):
            print("Starting Epoch:", epoch)

            # training loop
            running_loss = 0.0
            num_samples = 0
            
            self.model.train()
            for ibatch, batch in enumerate(tqdm(train_dataloader)):
                
                # run partial data based on input limit
                if ibatch < batch_max:
                    
                    # Reset the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(
                            pixel_values=batch["pixel_values"].to(device),
                            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                            class_labels=[labels.to(device) for labels in batch["class_labels"]],
                    )

                    # Backward propagation
                    loss = outputs.loss * loss_multiplier
                    loss.backward()

                    batch_size = batch["pixel_values"].size(0)
                    height = batch["pixel_values"].size(2)
                    width = batch["pixel_values"].size(3)
                    
                    running_loss += loss.item()
                    num_samples += batch_size
                    loss_train_cur = running_loss/num_samples

                    # Optimization
                    optimizer.step()
                    
                    # Update the learning rate at the specified epoch
                    if scheduler is not None:
                        scheduler.step()
                else:
                    # skip where ibatch >= batch_max
                    break

            # record loss at the end of each epoch
            logs['loss'] = loss_train_cur
            print( 'Epoch {:<4} training loss is: {:8.6f}.'.format(epoch, round(loss_train_cur, 6)) )
            loss_train.append(loss_train_cur)

            # validation loop
            running_loss = 0.0
            num_samples = 0
            
            self.model.eval()
            with torch.no_grad():
                for ibatch, batch in enumerate(tqdm(val_dataloader)):
                    # run partial data based on input limit
                    if ibatch < batch_max:
                        
                        # Forward pass
                        outputs = self.model(
                                pixel_values=batch["pixel_values"].to(device),
                                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                                class_labels=[labels.to(device) for labels in batch["class_labels"]],
                        )

                        # loss
                        loss = outputs.loss  * loss_multiplier

                        batch_size = batch["pixel_values"].size(0)
                        running_loss += loss.item()
                        num_samples += batch_size
                        loss_val_cur = running_loss/num_samples
                    else:
                        # skip where ibatch >= batch_max
                        break

            # record loss at the end of each epoch
            logs['val_loss'] = loss_val_cur
            
            print( 'Epoch {:<4} validation loss is: {:8.6f}.'.format(epoch, round(loss_val_cur, 6)) )
            loss_val.append(loss_val_cur)

            # if better model is found, update min_loss and save model (currently using training loss)
            if min_loss > loss_val_cur:
                print("Saved model in epoch",epoch)
                # save the best model
                self.data_handler.save_torch(file_name=f"model_current_{train_id}.pt", train_dir_prefix=train_dir_prefix, data=self.model)
                min_loss = loss_val_cur

            # save model regularly
            if epoch%self.save_interval == 0:
                print("Saved model in epoch",epoch)
                # save the best model
                self.data_handler.save_torch(file_name=f"model_epoch{epoch}_{train_id}.pt", train_dir_prefix=train_dir_prefix, data=self.model)

            # Update the plot with new logging information.
            liveloss.update(logs)
            liveloss.send()

        # save loss at the end
        self.data_handler.save_torch(file_name=f"losses_train_{train_id}.pt", train_dir_prefix=train_dir_prefix, data=loss_train)
        self.data_handler.save_torch(file_name=f"losses_val_{train_id}.pt", train_dir_prefix=train_dir_prefix, data=loss_val)
        
        # save default settings for record keeping
        self.data_handler.save_torch(file_name=f"train_settings_{train_id}.pt", train_dir_prefix=train_dir_prefix,
                                        data={"n_epoch": self.n_epoch,
                                              "resolution": (height, width),
                                              "optimizer": optimizer_state,
                                              "scheduler": scheduler_state,
                                              "batch_size": batch_size})
        
        torch.cuda.empty_cache()