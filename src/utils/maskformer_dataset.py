import numpy as np
import random
import torch
import torchvision.transforms.functional as TF

from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from transformers import MaskFormerImageProcessor
from typing import List, Literal

from utils.data_handler import DataHandler, StructuralScan, LatentVector

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# useful reference:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://blog.paperspace.com/dataloaders-abstractions-pytorch/

def collate_fn(batch):
    """
    Custom collate function used by torch.utils.data.DataLoader class
    Defines a collation dictionary mapping the different entities we track
    through the batches as data flows through MaskFormer.

    Inputs:
        batch - a batch of data from MaskformerMRIDataset of size batch size

    Returns a mapping dictionary
    """
    return {
            "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
            "pixel_mask": torch.stack([example["pixel_mask"] for example in batch]),
            "class_labels": [example["class_labels"] for example in batch],
            "mask_labels": [example["mask_labels"] for example in batch],
            "subj_no": [example["subj_no"] for example in batch],
            "file_no": [example["file_no"] for example in batch]
            }

# Dataloader for the MaskFormer MRI dataset
class MaskformerMRIDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self,  
                 data_list: List[str], 
                 data_dir: str,
                 processor: MaskFormerImageProcessor,
                 data_type: Literal["regular", "latent_space_vector"] = "regular", 
                 transform=None, 
                 augment=None):
        """
        Args:
            dataset
        """
        # Valid types: 
        self.data_dir = data_dir
        self.processor = processor
        self.transform = transform
        self.augment = augment

        # identify what type of data is coming in
        assert data_type == "regular" or data_type == "latent_space_vector", f"{data_type} is not a valid option."
        self.data_type = data_type
        # based on data_type, we can set what to expect for the channels
        if self.data_type == "regular":
            # regular structural scans
            self.channel_1 = StructuralScan.FLAIR
            self.channel_2 = StructuralScan.T1
            self.channel_3 = StructuralScan.T1GD
        else:
            # latent space vectors!
            self.channel_1 = LatentVector.LATENT_VECTOR_1
            self.channel_2 = LatentVector.LATENT_VECTOR_2
            self.channel_3 = LatentVector.LATENT_VECTOR_3

        # use the Data Handler class to handle all sorts of image loading
        self.data_handler = DataHandler()

        # expected format: UPENN-GBM-00008_53.nii.gz, UPENN-GBM-00008_54.nii.gz, UPENN-GBM-00008_55.nii.gz
        self.data_list = data_list.copy()
        self.n_data = len(self.data_list)

    def __len__(self):
        """
        custom definition of __len__ method to return class attribute self.n_data
        """
        return self.n_data

    def __getitem__(self, idx):
        """
        custom defintion of __getitem__ method defining how to iterate over custom
        dataset
        """

        if idx >= self.n_data:
            print("warning: given index",idx,"does not exist in data. Using firs sample instead.")

        # find a file corresponding to idx
        item = None
        try:
          item = self.data_list[idx]
        except IndexError:
          item = self.data_list[0]

        subj_no = item.split('.')[0].split('_')[0]
        file_no = item.split('.')[0].split('_')[1]
        
        # load channel 1
        data_cur, data_cur_nifti = self.data_handler.load_mri(
            subj_id=subj_no,
            file_no=file_no,
            struct_scan=self.channel_1,
            return_nifti=True,
            local_path=self.data_dir
        )

        # print(data_cur.shape)
        n_h = data_cur_nifti.shape[0]
        n_w = data_cur_nifti.shape[1]
        image = np.zeros( (n_h, n_w, 3) )

        # convert data range from [0 1] to [0 255]
        image[:,:,0] = data_cur * 255

        # load channel 2
        data_cur = self.data_handler.load_mri(
            subj_id=subj_no,
            file_no=file_no,
            struct_scan=self.channel_2,
            local_path=self.data_dir
        )
        image[:,:,1] = data_cur * 255

        # load channel 3
        data_cur = self.data_handler.load_mri(
            subj_id=subj_no,
            file_no=file_no,
            struct_scan=self.channel_3,
            local_path=self.data_dir
        )
        image[:,:,2] = data_cur * 255

        # load segm file
        data_cur = self.data_handler.load_mri(
            subj_id=subj_no,
            file_no=file_no,
            dtype=np.uint8,
            local_path=self.data_dir,
        )
        instance_seg =  np.zeros( (n_h, n_w), dtype=np.uint8)
        instance_seg[:,:] = data_cur
        
        # currently set mapping manually
        mapping_dict = {}
        mapping_dict[0] = 0
        mapping_dict[1] = 1
        mapping_dict[2] = 2
        mapping_dict[4] = 3

        # Use NumPy's vectorize() function to apply the mapping function to each element in the original array
        class_id_map = np.vectorize(lambda x: mapping_dict[x])(instance_seg)
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply data augmentation
        if self.augment is True:

            # Image Color Jittering
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # tf
            # color_jitter = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            # pil_image = color_jitter(pil_image)

            # apply contrast
            if random.random() <= 0.5:
                gamma_factor = 1.6
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(gamma_factor)

            # convert back to numpy
            image = np.array(pil_image)

            # convert to C, H, W (torchvision transforms assume this shape)
            image = image.transpose(2,0,1)
            n1_tmp = instance_seg.shape[0]
            n2_tmp = instance_seg.shape[1]
            instance_seg = instance_seg.reshape((1,n1_tmp,n2_tmp))

            # convert to tensors
            image = torch.from_numpy(image.astype('float'))
            instance_seg = torch.from_numpy(instance_seg)

            # Apply random horizontal flip to image and mask
            if random.random() <= 0.3:
                image = TF.hflip(image)
                instance_seg = TF.hflip(instance_seg)

            # Apply random crop to both the image and mask (as tensors)
            factor1 = 0.8
            if random.random() <= 0.2:
                chance1 = random.choice([0,1,2,3,4])
                dim1 = image.shape
                # print('Original image dimension:',dim1,'choice:',chance1)
                if chance1 == 0: #upper left
                    image        =        image[:,:int(dim1[1]*factor1),:int(dim1[2]*factor1)]
                    instance_seg = instance_seg[:,:int(dim1[1]*factor1),:int(dim1[2]*factor1)]
                elif chance1 == 1: #upper right
                    image        =        image[:,:int(dim1[1]*factor1),int(dim1[2]*(1-factor1)):]
                    instance_seg = instance_seg[:,:int(dim1[1]*factor1),int(dim1[2]*(1-factor1)):]
                elif chance1 == 2: #lower right
                    image        =        image[:,int(dim1[1]*(1-factor1)):,int(dim1[2]*(1-factor1)):]
                    instance_seg = instance_seg[:,int(dim1[1]*(1-factor1)):,int(dim1[2]*(1-factor1)):]
                elif chance1 == 3: #lower left
                    image        =        image[:,int(dim1[1]*(1-factor1)):,:int(dim1[2]*factor1)]
                    instance_seg = instance_seg[:,int(dim1[1]*(1-factor1)):,:int(dim1[2]*factor1)]
                else: # center
                    image        =        image[:,int(dim1[1]*(1-factor1)*0.5):int(dim1[1]*(1+factor1)*0.5), \
                                                  int(dim1[2]*(1-factor1)*0.5):int(dim1[2]*(1+factor1)*0.5)]
                    instance_seg = instance_seg[:,int(dim1[1]*(1-factor1)*0.5):int(dim1[1]*(1+factor1)*0.5), \
                                                  int(dim1[2]*(1-factor1)*0.5):int(dim1[2]*(1+factor1)*0.5)]

            # change back to ndarray
            image = image.numpy()
            instance_seg = instance_seg.numpy()
            instance_seg = instance_seg[0,:,:]

            # convert to H, W, C (transform requires this)
            image = image.transpose(1,2,0)

        # apply input transforms, including resize (after cropping)
        if self.transform:
            transformed = self.transform(image=image, mask=instance_seg)

            image, instance_seg = transformed['image'], transformed['mask']

        # Prepare data to fit Maskformer input
        inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        # add file and subj to input
        inputs["subj_no"] = subj_no
        inputs["file_no"] = file_no

        return inputs
