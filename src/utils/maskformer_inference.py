'''Functions used for inferencing'''

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

def get_mask_from_segm(segm_result: List[torch.Tensor]):
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

    # output is predicted mask with shape: (n, 512, 512)
    return pred_mask_labels

class MaskFormerInference():
    def __init__(self, data_handler: DataHandler, model: MaskFormerModel, processor: MaskFormerImageProcessor, transform: Compose):
        self.model = model
        self.data_handler = data_handler
        self.processor = processor
        self.transform = transform
        
    def predict_3d_mask(self, data_list:List[str], data_identifier: MriType):
        
        # define dataset given provided list
        dataset = MaskformerMRIDataset(data_handler=self.data_handler, 
                                       data_identifier=data_identifier,
                                       data_list=data_list, processor=self.processor,
                                       transform=self.transform, augment=False)

        # define data loader
        batch_size = 1
        metric_dataloader = DataLoader(dataset, batch_size=batch_size,
                                        shuffle=False, collate_fn=collate_fn)

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
        with torch.no_grad():
            for ibatch, batch in enumerate(metric_dataloader):

                # forward pass
                output_cur = self.model(
                        pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                        class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )

                # get first item in batch where batch size = 1
                image_cur = batch["pixel_values"][0]
                image_3d[ibatch, :, :, :] = image_cur.numpy()

                # post-processing/inference
                result_cur = self.processor.post_process_instance_segmentation(output_cur,
                                                                        target_sizes=[transforms.ToPILImage()(image_cur).size[::-1]])[0]
                
                # e.g. (n, 512, 512) where n is the number of existing labels in the prediction
                pred_class_labels = set([item["label_id"] for item in result_cur['segments_info']])
                pred_class_labels = sorted(list(pred_class_labels))
                pred_mask_labels = get_mask_from_segm(segm_result=result_cur)

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

        return image_3d, mask_true_3d, mask_pred_3d, all_true_labels, all_pred_labels