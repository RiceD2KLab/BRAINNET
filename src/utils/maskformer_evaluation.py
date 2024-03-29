'''Functions used for calculating segmentation metrics'''

import joblib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import seaborn as sns
from typing import List

import utils.maskformer_utils as mf_utils
import utils.metrics as metrics
import utils.mri_common as mri_common
from utils.mri_plotter import MRIPlotter
from utils.data_handler import DataHandler
from utils.metrics import MetricName

def calc_slice_metrics(all_labels, id2label, input_mask_labels, pred_mask_labels, input_class_labels, pred_class_labels):
    """
    Helper function to print out metrics for 2d predicted mask

    Inputs:
        batch - a batch of data from MaskformerMRIDataset of size batch size

    Returns a mapping dictionary
    """
    mask_shape = pred_mask_labels[0, :, :].shape

    # loop through segments 0 to 3
    for label_id in all_labels:

        print(f"Metrics for {label_id} - {id2label[label_id]}:")
        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
        true_mask = np.zeros(mask_shape, dtype=np.uint8)

        if label_id in input_class_labels:
            # extract layer in input_mask_labels corresponding to the label
            true_mask_idx = input_class_labels.index(label_id)
            true_mask = input_mask_labels[true_mask_idx, :, :].numpy().squeeze().astype(np.uint8)

        if label_id in pred_class_labels:
            # extract layer in pred_mask_labels corresponding to the label
            pred_mask_idx = pred_class_labels.index(label_id)
            pred_scaled_mask = pred_mask_labels[pred_mask_idx, :, :].squeeze()
            pred_mask = mf_utils.descale_mask(pred_scaled_mask)

        # calculate metrics
        dice_score = metrics.calc_dice_score(pred_mask, true_mask)
        print("Dice Score", dice_score)

        hd95 = metrics.calc_hausdorff_95(pred_mask, true_mask)
        print("HD95", hd95)

        other_metrics = metrics.calc_binary_metrics(pred_mask, true_mask)
        print("Specificity", other_metrics["specificity"])
        print("Sensitivity", other_metrics["sensitivity"])
        print("Precision", other_metrics["precision"])
        print("Recall", other_metrics["recall"])
        print()

# created outside of the class to avoid threading issue
def calc_metric_all_segments_pool(label_names, metric_func, mask_pred_3d, mask_true_3d):
    """
    Private class method to parallelize computation of hausdorff distance

    Args:
        label_names (list): list of ints representing segmentation mask label names
        metric_func (func): a metrics function from utils.metrics
        mask_pred_3d (numpy.ndarray): predicted segmentation mask
        mask_true_3d (numpy.ndarray): ground truth segmentation mask

    Returns:
        list of floats
    """
    # Create a pool of worker processes
    pool = multiprocessing.Pool()
    args_list = [(label_idx, metric_func, mask_pred_3d, mask_true_3d) for label_idx in range(len(label_names))]
    results = pool.map(_calc_metric_single_segment, args_list)

    # Close the pool to free resources
    pool.close()

    # Wait for all processes to complete
    pool.join()

    return results

def calc_metric_all_segments(label_names, metric_func, mask_pred_3d, mask_true_3d):
    """
    Private class method to compute metrics for all segmentation labels

    Args:
        label_names (list): list of ints representing segmentation maksk label names
        metric_func (func): a metrics function from utils.metrics
        mask_pred_3d (numpy.ndarray): predicted segmentation mask
        mask_true_3d (numpy.ndarray): ground truth segmentation mask

    Returns:
        list of floats
    """
    results = []
    for label_idx in range(len(label_names)):
        true_mask_cur = mask_true_3d[label_idx,:,:,:]
        pred_mask_cur = mask_pred_3d[label_idx,:,:,:]
        result = metric_func(pred_mask_cur, true_mask_cur)
        results.append(result)
    return results


# created outside of the class to avoid threading issue
def calc_subject_metrics(label_names, subj_id, mask_pred_binary, mask_true_binary):
    """
    Calculates metrics for a given subject using the predicted mask and
    ground truth mask.

    Metrics are:
    1) Dice Score
    2) 95th percentile Hausdorff Distance
    3) Common: Precision, Recall, Sensitivity, & Specificity

    Args:
        subj_id (str): The unique subject identifier (eg. UPENN-GBM-00312)
        mask_pred_binary (numpy.ndarray): predicted binary mask
        mask_true_binary (numpy.ndarray): ground truth binary mask

    Returns:
        tuple of float, float, and dictionary of "common" metrics
    """

    print("Calculating metrics for ", subj_id)
    # dice coefficient
    # output array of dice score for all segments: [0.9, 0.8, 0.92, 0.3]
    dice_score = calc_metric_all_segments(label_names, metrics.calc_dice_score, mask_pred_binary, mask_true_binary)

    # 95% hausdorff distance
    # output array of hd95 for all segments: [5.3, 2.8, 3.92, 1]
    hausdorff_val = calc_metric_all_segments_pool(label_names, metrics.calc_hausdorff_95, mask_pred_binary, mask_true_binary)

    # common metrics
    # e.g. 'true_positive': [20927582, 148267, 687623, 226880]
    common_metrics = calc_metric_all_segments(label_names, metrics.calc_binary_metrics, mask_pred_binary, mask_true_binary)
    common_metrics_dict = {}
    for key in common_metrics[0]:
        common_metrics_dict[key] = [metric[key] for metric in common_metrics]

    return dice_score, hausdorff_val, common_metrics_dict


def _calc_metric_single_segment(args):
    """
    Private class method to calculate metrics for a single segment label

    Args:
        label_idx (int): slice to calculate
        metric_func (func): a metrics functions from utils.metrics
        mask_pred_3d (numpy.ndarray): predicted segmentation mask
        mask_true_3d (numpy.ndarray): ground truth segmentation maks

    Returns
        float
    """
    label_idx, metric_func, mask_pred_3d, mask_true_3d = args
    true_mask_cur = mask_true_3d[label_idx, :, :, :]
    pred_mask_cur = mask_pred_3d[label_idx, :, :, :]
    result = metric_func(pred_mask_cur, true_mask_cur)
    return result
    
class MaskFormerEvaluation():
    def __init__(self):
        """
        Defines evaluation metrics and visualizations for use with MaskFormer
        """
        self.all_label_colors = list(mri_common.SEGMENT_COLORS.values())
        self.mri_plt = MRIPlotter()
        self.data_handler = DataHandler()
        self.all_label_names = list(mri_common.SEGMENTS.values())

    def predict_and_eval(self, mf_inference, subj_names: List[str], metrics_dir_prefix: str, metrics_file_name: str, recalculate: bool, batch_size=10):
        """
        Passes input images through trained MaskFormer to predict segmentation masks and then
        calculates metrics

        Args:
            mf_inference (MaskFormerInference): instance of MaskFormerInference class (train/val/test)
            subj_names (list): list of strings containing subject names
            metrics_dir_prefix (str): path to save metrics outputs
            metrics_file_name (str): output file name for saving metrics
            recalculate (bool): flag indicating whether to recalculate metrics
            batch_size (int): size of inputs to pass through in a single batch

        Returns:
            tuple of dictionary containing metrics, list of error files,
            and list of success files
        """

        all_dice = []
        all_hd95 = []
        all_common_metrics = []
        error_files = []

        metrics_summary = {}

        if recalculate:
                all_dice, all_hd95, all_common_metrics, error_files = self._predict_and_calc_metrics(mf_inference=mf_inference, subj_selected=subj_names, batch_size=batch_size)

                print("Files with error:", error_files)

                # remove files with error
                success_files = list(filter(lambda x: x not in error_files, subj_names))
                print(f"No. of files after removing problematic images: {len(success_files)}")

                # replace hd94 None or np.inf values with highest hd95
                all_hd95 = self._update_invalid_hd95(all_hd95)

                # get BraTs metrics from all_common_metrics
                all_precision = [metric[MetricName.PRECISION.value] for metric in all_common_metrics]
                all_recall = [metric[MetricName.RECALL.value] for metric in all_common_metrics]
                all_sensitivity = [metric[MetricName.SENSITIVITY.value] for metric in all_common_metrics]
                all_specificity = [metric[MetricName.SPECIFICITY.value] for metric in all_common_metrics]

                metric_scores = {
                    MetricName.PRECISION.value: np.array(all_precision),
                    MetricName.RECALL.value: np.array(all_recall),
                    MetricName.SENSITIVITY.value: np.array(all_sensitivity),
                    MetricName.SPECIFICITY.value: np.array(all_specificity),
                    MetricName.DICE_SCORE.value: np.array(all_dice),
                    MetricName.HD95.value: np.array(all_hd95)
                }

                metrics_summary = {
                    "scores": metric_scores,
                    "error_files": error_files,
                    "success_files": success_files
                }

                # dump results to temp file
                metrics_temp_file = self.data_handler.create_temp_file(metrics_file_name)
                joblib.dump(metrics_summary, metrics_temp_file)

                # upload to cloud
                self.data_handler.save_from_source_path(file_name=metrics_file_name, source_path=metrics_temp_file,
                                                train_dir_prefix=metrics_dir_prefix, use_cloud=True)
        else:
            # load metrics results from file if recalculate = False
            metrics_temp_file = self.data_handler.load_to_temp_file(file_name=metrics_file_name,
                                                            train_dir_prefix=metrics_dir_prefix)

            metrics_summary = joblib.load(metrics_temp_file)

        print("Files with error:", metrics_summary["error_files"])
        print("Total Files evaluated: " + str(len(metrics_summary["success_files"])))

        return metrics_summary["scores"], metrics_summary["error_files"], metrics_summary["success_files"]

    def _update_invalid_hd95(self, all_hd95):
        """
        Private class method to set default max possible hausdorff distance
        max value is specified as 373.12866 per Brats 2021 challenge winner.

        Args:
            all_hd95 (list): list of hd95 scores as floats

        Returns:
            a list of floats. Does not mutate input.
        """
        all_hd95_copy = all_hd95.copy()
        max_value = 0
        none_indices = []
        for subj_idx, hd95_per_subj in enumerate(all_hd95):
            for segm_idx, hd95 in enumerate(hd95_per_subj):
                if hd95 is None or hd95 == np.inf:
                    # get all indices with invalid values
                    none_indices.append((subj_idx, segm_idx))

        # indicated default number in 373.12866 in Brats 2021 literature
        max_value = 373.12866
        print("indices with invalid hd95", str(none_indices))
        
        # replace the invalid values with max hd value
        for row, col in none_indices:
            all_hd95_copy[row][col] = max_value
        return all_hd95_copy

    def _predict_and_calc_metrics(self, mf_inference, subj_selected: List[str], batch_size: int):
        """
        Private class method to generate predicted masks and calculate metrics

        Args:
            mf_inference (MaskFormerInference): instance of MaskFormerInference class (train/val/test)
            subj_selected (list): list of strings of selected subjects
            batch_size (int): batch size to use in processing through predict and calc

        Returns:
            tuple of lists
        """
        all_dice = []
        all_hd95 = []
        all_common_metrics = []
        error_files = []
        mask_true_3d, mask_pred_3d = None, None

        for subj_id in subj_selected:

            # perform inference for each slice within the volume
            mask_3d_pred_results = mf_inference.predict_patient_mask(subj_id=subj_id, batch_size=batch_size)
            mask_true_3d, mask_pred_3d = mask_3d_pred_results[1], mask_3d_pred_results[2]

            # since prediction range is from 0 to 255, convert data back to binary
            mask_pred_3d_binary = mf_utils.descale_mask(mask_pred_3d)
            mask_true_3d_binary = mf_utils.descale_mask(mask_true_3d)

            try:
                dice_score, hausdorff_val, common_metrics_dict = calc_subject_metrics(
                    label_names=self.all_label_names, subj_id=subj_id,
                    mask_pred_binary=mask_pred_3d_binary, mask_true_binary=mask_true_3d_binary)

                # append only when there is no error
                all_dice.append(dice_score)
                all_hd95.append(hausdorff_val)
                all_common_metrics.append(common_metrics_dict)

            except Exception as ex:
                print(f"Error {subj_id}", ex)
                error_files.append(subj_id)

        return  all_dice, all_hd95, all_common_metrics, error_files


    def get_mean_scores(self, metrics_dict):
        """
        Calculates the mean of each metric for all samples

        Args:
            metrics_dict (dict): dictionary containing metrics where key is
            metric name and value is numpy array of scores for each sample

        Returns:
            pandas.DataFrame of metric mean values
        """
        metric_names = [
            MetricName.DICE_SCORE.value,
            MetricName.HD95.value,
            MetricName.SENSITIVITY.value,
            MetricName.SPECIFICITY.value
        ]
        mean_scores = {}
        for metric_name in metric_names:
            mean_scores[metric_name] = list(metrics_dict[metric_name].mean(axis=0))

        # Create a DataFrame from the list
        return pd.DataFrame(mean_scores.values(), index=list(mean_scores.keys()),  columns=self.all_label_names)

    # metric plotting functions
    def display_statistics(self, metrics_dict):
        """
        Constructs a summary statistics table for each metric

        Args:
            metrics_dict (dict): dictionary containing metrics where key is
            metric name and value is numpy array of scores for each sample

        Returns:
            None
        """
        for metric_name, metric_scores in metrics_dict.items():
            print("Summary:", metric_name)
            df = pd.DataFrame(metric_scores, columns=self.all_label_names)
            print(df.describe())
            print("\n")

    def draw_box_plots(self, metrics_dict, set_ylim=False):
        """
        Plots metric statistics as a box plot for each segmentation label

        Args:
            metrics_dict (dict): dictionary containing metrics where key is
            metric name and value is numpy array of scores for each sample
            set_ylim (bool): flag to indicate whether to set ylim explicity to 1

        Returns:
            None
        """
        nrows = 3
        ncols = 2

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 20))

        count = 0
        for metric_name, metric_scores in metrics_dict.items():
            row = count//ncols
            col = count%ncols
            ax = axs[row, col]

            sns.boxplot(data=metric_scores, palette=self.all_label_colors, ax=ax)
            ax.set_xticks(range(len(self.all_label_names)), self.all_label_names)
            ax.set_ylabel("score")
            ax.set_title(metric_name)
            if set_ylim:
                ax.set_ylim(0, 1.02)

            count+=1
            ax.grid(True)
            self.mri_plt.set_axis_config(ax)

        plt.show()

    def build_histogram(self, metrics_dict):
        """
        Plots metrics statistics as a histogram colored by each segmentation label

        Args:
            metrics_dict (dict): dictionary containing metrics where key is
            metric name and value is numpy array of scores for each sample

        Returns:
            None
        """
        nrows = 3
        ncols = 2

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 20))

        count = 0

        # for each metric
        for metric_name, scores in metrics_dict.items():
            row = count//ncols
            col = count%ncols
            ax = axs[row, col]

            x0 = scores[:, 0]
            x2 = scores[:, 1]
            x3 = scores[:, 2]
            x4 = scores[:, 3]
            ax.hist([x0, x2, x3, x4], bins = 20, label=self.all_label_names, color=self.all_label_colors)
            ax.legend()
            ax.set_xlabel(metric_name)
            ax.set_ylabel('count')
            ax.grid(True)

            self.mri_plt.set_axis_config(ax)

            count+=1

        plt.show()


    def get_low_dice_score_images(self, subj_list, summary_metrics_dict, top_n=10):
        """
        Identifies samples with lowest dice scores for QC

        Args:
            subj_list (list): list of strings containing unique subject id's
            summary_metrics_dict (dict): dictionary containing metrics where key is
            metric name and value is numpy array of scores for each sample
            top_n (int): how many records to return

        Returns:
            pandas.DataFrame of samples with top_n lowest Dice scores.
        """
        flattened_array = summary_metrics_dict[MetricName.DICE_SCORE.value].flatten()

        # rearranges the elements such that the smallest n elements come before the nth index
        top_n_idx_1d = np.argpartition(flattened_array, kth=top_n)[:top_n]

        # map the 1d indices to their corresponding 2D indices to match with the correct row (dataset_idx) and col (label_id affected)
        top_n_indices = np.unravel_index(top_n_idx_1d, summary_metrics_dict[MetricName.DICE_SCORE.value].shape)

        # use the indices to retrieve the actual values
        top_n_worst_scores = summary_metrics_dict[MetricName.DICE_SCORE.value][top_n_indices]
        dataset_idx = top_n_indices[0]
        label_id = top_n_indices[1]
        subj_names = np.array(subj_list)[dataset_idx]

        # create a dictionary
        top_n_worst_dice= []
        for subject, dataset_idx, label_id, score in zip(subj_names, dataset_idx, label_id, top_n_worst_scores):
            top_n_worst_dice.append(
                {
                    'subj_name': subject,
                    'dataset_idx': dataset_idx,
                    'label_id': label_id,
                    'score': score
                })

        # sort and convert to dataframe
        top_n_worst_dice_sorted = sorted(top_n_worst_dice, key=lambda x: x['score'])
        return pd.DataFrame(top_n_worst_dice_sorted)


    def get_high_hd95_images(self, subj_list, summary_metrics_dict, top_n=10):
        """
        Identifies samples with highest 95th percentile Hausdorff distance for QC

        Args:
            subj_list (list): list of strings containing unique subject id's
            summary_metrics_dict (dict): dictionary containing metrics where key is
            metric name and value is numpy array of scores for each sample
            top_n (int): how many records to return

        Returns:
            pandas.DataFrame of samples with top_n highest 95th perc. Hausdorff distance
        """
        flattened_array = summary_metrics_dict[MetricName.HD95.value].flatten()

        # rearranges the elements such that the smallest n elements come before the nth index
        top_n_idx_1d = np.argpartition(flattened_array, kth=(-1*top_n))[(-1*top_n):]

        # map the 1d indices to their corresponding 2D indices to match with the correct row (dataset_idx) and col (label_id affected)
        top_n_indices = np.unravel_index(top_n_idx_1d, summary_metrics_dict[MetricName.HD95.value].shape)

        # use the indices to retrieve the actual values
        top_n_worst_scores = summary_metrics_dict[MetricName.HD95.value][top_n_indices]
        dataset_idx = top_n_indices[0]
        label_id = top_n_indices[1]
        subj_names = np.array(subj_list)[dataset_idx]

        top_n_worst_hd95= []
        for subject, dataset_idx, label_id, score in zip(subj_names, dataset_idx, label_id, top_n_worst_scores):
            top_n_worst_hd95.append(
                {
                    'subj_name': subject,
                    'dataset_idx': dataset_idx,
                    'label_id': label_id,
                    'score': score
                })
        top_n_worst_hd95_sorted = sorted(top_n_worst_hd95, key=lambda x: (-1*x['score']))
        return pd.DataFrame(top_n_worst_hd95_sorted)