# All functions used for inferencing

import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from transformers import (MaskFormerImageProcessor, MaskFormerModel)
from typing import List
from albumentations import Compose

# local imports
import utils.maskformer_utils as mf_utils
import utils.mri_common as mri_common

from utils.data_handler import DataHandler, MriType
from utils.maskformer_dataset import MaskformerMRIDataset, collate_fn

def get_mask_from_segm_result(segm_result: List[torch.Tensor]):
    # extracts prediction from post_process_semantic_segmentation result
    pred_class_labels = set([item["label_id"] for item in segm_result['segments_info']])
    pred_class_labels = sorted(list(pred_class_labels))

    pred_mask_shape = (len(pred_class_labels), segm_result['segmentation'].shape[0], segm_result['segmentation'].shape[1])
    pred_mask_labels = np.zeros(pred_mask_shape, dtype=np.uint8)

    # get predicted mask for each segment
    for mask_idx, mask_id in enumerate(pred_class_labels):
        mask_pred_2d = np.zeros((pred_mask_shape[1], pred_mask_shape[2]), dtype=np.uint8)

        # find predicted masks in results
        for item in segm_result['segments_info']:
            if item['label_id'] == mask_id:
                # get mask will scale to 255
                mask_pred_2d += np.array( mf_utils.get_mask(segm_result['segmentation'], item['id']) )

        mask_pred_2d = mask_pred_2d.astype(np.uint8)
        pred_mask_labels[mask_idx, :, :] = mask_pred_2d

    # output is predicted mask with shape: (num_labels, height, width)
    return pred_mask_labels, pred_class_labels
    
class MaskFormerInference_upscaled():
    '''Assumes that the data reside in one folder'''
    
    def __init__(self, data_handler: DataHandler, data_identifier: MriType, model: MaskFormerModel, processor: MaskFormerImageProcessor, transform: Compose, transform2: Compose = None, orig_dim: tuple = None):
        self.model = model
        self.data_handler = data_handler
        self.processor = processor
        self.transform = transform
        self.transform2 = transform2
        self.data_identifier = data_identifier
        self.orig_dim = orig_dim
    
        # UPENN-GBM-00006_11_FLAIR_1.nii.gz, UPENN-GBM-00006_11_T1_1.nii.gz, UPENN-GBM-00006_11_FLAIR_2.nii.gz...
        all_files_in_dir = self.data_handler.list_mri_in_dir(mri_type=data_identifier)
        
        # UPENN-GBM-00040_122.nii.gz, UPENN-GBM-00073_20.nii.gz, UPENN-GBM-00016_139.nii.gz
        self.all_files = list(set([mri_common.get_mri_slice_file_name(file_name) for file_name in all_files_in_dir]))
        
    def get_patient_slices(self, subj: str):
        # if substring/subj is in the filename, include that file in the list of slices
        vol_list = [file_name for file_name in self.all_files if subj in file_name]
        
        # sort slices: UPENN-GBM-00073_1.nii.gz, UPENN-GBM-00073_2.nii.gz, UPENN-GBM-00073_3.nii.gz
        vol_list_sorted = sorted(vol_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return vol_list_sorted
        
    def predict_segm(self, batch):
        first_img =  batch["pixel_values"][0]
        target_size = transforms.ToPILImage()(first_img).size[::-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(
                            pixel_values=batch["pixel_values"].to(device),
                            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                            class_labels=[labels.to(device) for labels in batch["class_labels"]],
                    )
            
            torch.cuda.empty_cache()
            
        # post-processing/inference
        results=self.processor.post_process_instance_segmentation(outputs, target_sizes=[target_size])[0]
        return results
    
    def predict_patient_mask(self, subj_id: str):
        
        print("Performing inference on", subj_id)
        # get all slices for one patient or subj: expected is 146
        data_list = self.get_patient_slices(subj_id)  

        # create two datasets: one with original transform, another with upscaled transform
        dataset = MaskformerMRIDataset(data_handler=self.data_handler, 
                                       data_identifier=self.data_identifier,
                                       data_list=data_list, processor=self.processor,
                                       transform=self.transform, augment=False)
        dataset_upscale = MaskformerMRIDataset(data_handler=self.data_handler, 
                                       data_identifier=self.data_identifier,
                                       data_list=data_list, processor=self.processor,
                                       transform=self.transform2, augment=False)

        # define data loader
        # TODO: review if batch_size is fixed
        batch_size = 1
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        dataloader_upscale = DataLoader(dataset_upscale, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # initalize 3d variables using shape of first input
        first_img_shape = dataset[0]["pixel_values"].shape

        # image_3d: (146, 3, 512, 512)
        num_slices = len(data_list)
        image_3d = np.zeros((num_slices, first_img_shape[0], first_img_shape[1], first_img_shape[2]), dtype=np.uint8)

        # true and predicted masks: (4, 146, 512, 512)
        mask_shape_3d = (len(mri_common.SEGMENTS), num_slices, first_img_shape[1], first_img_shape[2])
        mask_pred_3d = np.zeros(mask_shape_3d, dtype=np.uint8)
        mask_true_3d = np.zeros(mask_shape_3d, dtype=np.uint8)
        all_true_labels = []
        all_pred_labels = []

        # perform prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.model.eval()
        for (ibatch, batch), (_, batch_upscale) in zip(enumerate(dataloader), enumerate(dataloader_upscale)):

            with torch.no_grad():

                # post-processing/segmentation inference
                segm_result = self.predict_segm(batch_upscale)
                
                # e.g. (n, 512, 512) where n is the number of existing labels in the prediction
                pred_mask_labels, pred_class_labels = get_mask_from_segm_result(segm_result=segm_result)
                pred_mask_labels = mf_utils.resize_mask(pred_mask_labels, original_size=self.orig_dim)

                # build mask_pred_3d:
                for mask_idx, label in enumerate(pred_class_labels):
                    mask_pred_3d[label, ibatch, :, :] = pred_mask_labels[mask_idx, :, :]

                # obtain true mask where available
                true_mask_labels = batch["mask_labels"][0]
                true_class_labels = batch["class_labels"][0]
                for mask_idx, label in enumerate(true_class_labels.tolist()):
                    unscaled_mask = true_mask_labels[mask_idx,:,:]
                    mask_true_3d[label, ibatch, :, :] = mf_utils.scale_mask(unscaled_mask)

                all_true_labels = list(set(all_true_labels + true_class_labels.tolist()))
                all_pred_labels = list(set(all_pred_labels + pred_class_labels))
        
            torch.cuda.empty_cache()

        return image_3d, mask_true_3d, mask_pred_3d, all_true_labels, all_pred_labels