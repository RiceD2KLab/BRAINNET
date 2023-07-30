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
    segments_info = segm_result['segments_info']
    segmentation = segm_result['segmentation'].cpu().numpy()
    
    pred_class_labels = set([item["label_id"] for item in segments_info])
    pred_class_labels = sorted(list(pred_class_labels))
    pred_mask_shape = (len(pred_class_labels), segmentation.shape[0], segmentation.shape[1])
    pred_mask_labels = np.zeros(pred_mask_shape, dtype=np.uint8)

    # get predicted mask for each segment
    for mask_idx, pred_label in enumerate(pred_class_labels):
        mask_pred_2d = np.zeros((pred_mask_shape[1], pred_mask_shape[2]), dtype=np.uint8)

        # need to loop through segments_info because label_id with different instances can be found
        for item in segments_info:
            if item['label_id'] == pred_label:
                # get corresponding mask using item['id'] which is the instance value for that label
                mask = (segmentation  == item['id'])
                scaled_mask = mf_utils.scale_mask(mask)
                mask_pred_2d += np.array(scaled_mask)

        mask_pred_2d = mask_pred_2d.astype(np.uint8)
        pred_mask_labels[mask_idx, :, :] = mask_pred_2d

    # output is predicted mask with shape: (num_labels, height, width)
    return pred_mask_labels, pred_class_labels
    
class MaskFormerInference():
    
    def __init__(self, data_handler: DataHandler, data_identifier: MriType, model: MaskFormerModel, processor: MaskFormerImageProcessor, 
                 upscaled_transform: Compose, scale_to_orig_size=True, orig_transform: Compose = None, orig_dim: tuple = None):
        
        if scale_to_orig_size == True:
            assert orig_transform is not None, "provide orig_transform with unscaled image. otherwise, set 'scale_to_orig_size' param to false"
            assert orig_dim is not None, "provide tuple for with the original dimension of the image. otherwise, set 'scale_to_orig_size' param to false"
        
        self.model = model
        self.data_handler = data_handler
        self.processor = processor
        self.scale_to_orig_size = scale_to_orig_size
        
        # specifies which dataset to use
        # eg: train depth, train cross side, val cross front
        self.data_identifier = data_identifier
        
        # no resize transform
        self.orig_transform = orig_transform
        
        # original dimensions of the file
        self.orig_dim = orig_dim
        
        # predict transform
        self.upscaled_transform = upscaled_transform
    
        # UPENN-GBM-00006_11_FLAIR_1.nii.gz, UPENN-GBM-00006_11_T1_1.nii.gz, UPENN-GBM-00006_11_FLAIR_2.nii.gz...
        all_files_in_dir = self.data_handler.list_mri_in_dir(mri_type=data_identifier)
        
        # Just re-write to a format that the Maskformer DataLoader wants
        # UPENN-GBM-00040_122.nii.gz, UPENN-GBM-00073_20.nii.gz, UPENN-GBM-00016_139.nii.gz
        self.all_files = list(set([mri_common.get_mri_slice_file_name(file_name) for file_name in all_files_in_dir]))
        
    def get_patient_slices(self, subj: str):
        # if substring/subj is in the filename, include that file in the list of slices
        vol_list = [file_name for file_name in self.all_files if subj in file_name]
        
        # sort slices: UPENN-GBM-00073_1.nii.gz, UPENN-GBM-00073_2.nii.gz, UPENN-GBM-00073_3.nii.gz
        vol_list_sorted = sorted(vol_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return vol_list_sorted
    
    def predict_segm(self, batch, batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            model_outputs, results = self._predict_segm(batch, batch_size)
        torch.cuda.empty_cache()
        return model_outputs, results
         
    def predict_patient_mask(self, subj_id: str, batch_size=10):
        
        print("Performing inference on", subj_id)
        print("batch size", batch_size)
        
        # Get all slices for one patient or subj
        data_list = self.get_patient_slices(subj_id)  
        num_slices = len(data_list)
        batch_size = batch_size
        
        # dataloader for upscaled dataset
        dataset_upscaled = MaskformerMRIDataset(data_handler=self.data_handler, 
                                       data_identifier=self.data_identifier,
                                       data_list=data_list, 
                                       processor=self.processor,
                                       transform=self.upscaled_transform, 
                                       augment=False)
        dataloader_upscale = DataLoader(dataset_upscaled, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # initalize 3d variables using shape of first input
        first_img_shape = dataset_upscaled[0]["pixel_values"].shape
        
        # dataloader for dataset with original size
        dataloader_orig = None
        if self.scale_to_orig_size:
            dataset_orig = MaskformerMRIDataset(data_handler=self.data_handler, 
                                        data_identifier=self.data_identifier,
                                        data_list=data_list, 
                                        processor=self.processor,
                                        transform=self.orig_transform, 
                                        augment=False)
            dataloader_orig = DataLoader(dataset_orig, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            first_img_shape = dataset_orig[0]["pixel_values"].shape
        else:
            # create dummy downloader for looping purposes
            batch_count = len(dataloader_upscale)
            dataloader_orig = list(range(batch_count))
        
        # image_3d: (num_slices, 3 channels, width, height)
        image_3d = np.zeros((num_slices, first_img_shape[0], first_img_shape[1], first_img_shape[2]), dtype=np.uint8)

        # initalize true and predicted masks: (4, num_slices, width, height)
        mask_shape_3d = (len(mri_common.SEGMENTS), num_slices, first_img_shape[1], first_img_shape[2])
        mask_pred_3d = np.zeros(mask_shape_3d, dtype=np.uint8)
        mask_true_3d = np.zeros(mask_shape_3d, dtype=np.uint8)
        all_true_labels = []
        all_pred_labels = []

        # perform prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            slice_idx = 0
            for (ibatch, batch_orig), (_, batch_upscale) in zip(enumerate(dataloader_orig), enumerate(dataloader_upscale)):
                # determine input image batch to use:
                input_batch = batch_orig if self.scale_to_orig_size else batch_upscale
                
                # perform inference
                # always use upscaled version for post-processing i.e the transform with the size used during training
                _, segm_results = self._predict_segm(batch_upscale, batch_size)
                
                for batch_idx, segm_result in enumerate(segm_results):
                    
                    # get first item in batch
                    image_cur = input_batch["pixel_values"][batch_idx]
                    image_3d[slice_idx, :, :, :] = image_cur.numpy()

                    # e.g. (num_labels, width, height)
                    pred_mask_labels, pred_class_labels = get_mask_from_segm_result(segm_result=segm_result)
                    
                    # resize if specified
                    if self.scale_to_orig_size:
                        pred_mask_labels = mf_utils.resize_mask(pred_mask_labels, original_size=self.orig_dim)

                    # update mask_pred_3d by stacking each 2d prediction
                    # get 2d predicted mask for each predicted class label
                    for mask_idx, pred_label in enumerate(pred_class_labels):
                        # note that the shape of pred_mask_labels is dependent on the number of labels predicted
                        # e.g. if only 2 segments are predicted, the shape is (2, height, width)
                        pred_mask = pred_mask_labels[mask_idx, :, :]

                        # meanwhile, the shape of mask_pred_3d is fixed with the total number of segments as initialized above
                        # we can use the predicted label to determine which channel to save the mask into
                        mask_pred_3d[pred_label, slice_idx, :, :] = pred_mask

                    # now we build the mask_true_3d 
                    true_mask_labels = input_batch["mask_labels"][batch_idx]
                    true_class_labels = input_batch["class_labels"][batch_idx].tolist()
                    
                    for mask_idx, label in enumerate(true_class_labels):
                        unscaled_mask = true_mask_labels[mask_idx,:,:]
                        
                        # this scales prediction to 0 to 255
                        mask_true_3d[label, slice_idx, :, :] = mf_utils.scale_mask(unscaled_mask)

                    all_true_labels = all_true_labels + true_class_labels
                    all_pred_labels = all_pred_labels + pred_class_labels
                    slice_idx += 1
        
        torch.cuda.empty_cache()

        # get unique labels
        all_true_labels = list(set(all_true_labels))
        all_pred_labels = list(set(all_pred_labels))
                    
        return image_3d, mask_true_3d, mask_pred_3d, all_true_labels, all_pred_labels
    
    def _predict_segm(self, batch, batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        first_img =  batch["pixel_values"][0]
        target_size = transforms.ToPILImage()(first_img).size[::-1]
        target_sizes = [target_size for _ in range(batch_size)]
        outputs = self.model(
                        pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                        class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )
        # post-processing/inference
        results=self.processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)
        return outputs, results