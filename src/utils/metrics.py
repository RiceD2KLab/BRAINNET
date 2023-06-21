import medpy.metric.binary as mdp
import numpy as np

from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import _ni_support

# region Commonly used metrics
def calc_binary_metrics(segm_pred, segm_true, segm_id=None):
    '''
    precision: tp / tp + fp => positive predicted value
    recall: tp / tp + fn => ratio of true positives to total positives in the data
    specificity = tn / (tn + fp) => ratio of true nagatives to total negatives in the data
    '''
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)

    recall = mdp.recall(mask_pred, mask_true)
    specificity = mdp.specificity(mask_pred, mask_true)
    metrics_dict = {
        "true_positive": np.count_nonzero(mask_pred & mask_true),
        "true_negative": np.count_nonzero(~mask_pred & ~mask_true),
        "false_positive": np.count_nonzero(mask_pred & ~mask_true),
        "false_negative": np.count_nonzero(~mask_pred & mask_true),
        "precision": mdp.precision(mask_pred, mask_true),
        "recall": recall,
        "specificity": specificity,
        "sensitivity": recall,
        "true_positve_rate": recall,
        "true_negative_rate": mdp.true_negative_rate(mask_pred, mask_true)
    } 
    return metrics_dict

def calc_dice_score(segm_pred, segm_true, segm_id=None):
    mask_true = segm_true
    mask_pred = segm_pred
    
    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)

    # perform element wise multiplication to get intersection
    intersection = np.sum(mask_pred * mask_true)
    dice_score = (2.0 * intersection) / (np.sum(mask_true) + np.sum(mask_pred))
    return dice_score
  
def calc_miou(mask_pred, mask_true):
    num_i = len( np.where(mask_pred * mask_true != 0)[0] )
    num_u = len( np.where(mask_pred + mask_true != 0)[0] )
    miou = num_i / num_u * 100
    return miou, num_u
  
# endregion

# region Boundary-based metrics
def calc_hausdorff_95(segm_pred, segm_true, segm_id=None):
    return _exec_hausdorff_95(segm_pred, segm_true, segm_id=segm_id, plot=False)

def plot_hausdorff_95(segm_pred, segm_true, segm_id=None):
    return _exec_hausdorff_95(segm_pred, segm_true, segm_id=segm_id, plot=True)

def _calc_surface_distances(mask_pred, mask_true, voxelspacing=None, connectivity=1):
  """
  The distances between the surface voxel of binary objects in result and their
  nearest partner surface voxel of a binary object in reference.
  
  Source Code: https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L357
  """
  if voxelspacing is not None:
      # ensures that the sequence has the same length as the number of dimensions in the result array
      # not needed for now
      voxelspacing = _ni_support._normalize_sequence(voxelspacing, mask_pred.ndim)
      voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
      if not voxelspacing.flags.contiguous:
          voxelspacing = voxelspacing.copy()
          
  # this will create 3x3 binary array with False values at the center
  # and its surrounding structures. diagonals are not considered if connectivity = 1
  footprint = generate_binary_structure(mask_pred.ndim, connectivity)

  # extract only 1-pixel border line of objects
  # applies binary erosion to the image
  segm_pred_border = mask_pred ^ binary_erosion(mask_pred, structure=footprint, iterations=1)
  segm_true_border = mask_true ^ binary_erosion(mask_true, structure=footprint, iterations=1)

  # calculate distances from each voxel to the closest border 
  dist = distance_transform_edt(~segm_true_border, sampling=voxelspacing)
  surface_dist = np.where(segm_pred_border, dist, None)
  return surface_dist

def _exec_hausdorff_95(segm_pred, segm_true, segm_id=None, plot=False):
    mask_true = segm_true
    mask_pred = segm_pred

    if segm_id is not None:
        mask_true = _get_binary_mask(segm_true, segm_id)
        mask_pred = _get_binary_mask(segm_pred, segm_id)
        
    if np.sum(mask_true) + np.sum(mask_pred) == 0:
        print("Both images do not have the segment ids")
        return None

    connectivity = mask_pred.ndim * 2
    surface_dist_pred = _calc_surface_distances(mask_pred, mask_true, connectivity=connectivity)
    surface_dist_true = _calc_surface_distances(mask_true, mask_pred, connectivity=connectivity)
    surface_dist_pred_no_null = surface_dist_pred[surface_dist_pred != None]
    surface_dist_true_no_null = surface_dist_true[surface_dist_true != None]
    hd95_val = np.percentile(np.hstack((surface_dist_pred_no_null, surface_dist_true_no_null)), 95)

    if plot:
        return hd95_val, surface_dist_pred, surface_dist_true
    else:
        return hd95_val

# endregion

# region Helper Functions
def _get_binary_mask(img, segm_id):
    return img == segm_id
# endregion