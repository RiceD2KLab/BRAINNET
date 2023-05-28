import medpy.metric.binary as mdp
import numpy as np

from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import _ni_support

# region Commonly used metrics
def calc_binary_metrics(pred_img, true_img, segment):
    '''
    precision: tp / tp + fp => positive predicted value
    recall: tp / tp + fn => ratio of true positives to total positives in the data
    Specificity = tn / (tn + fp) => ratio of true nagatives to total negatives in the data
    '''
    
    true_img_bin = _convert_to_binary(true_img, segment)
    pred_img_bin = _convert_to_binary(pred_img, segment)

    recall = mdp.recall(pred_img_bin, true_img_bin)
    
    return {
        "true_positive": np.count_nonzero(pred_img & true_img),
        "true_negative": np.count_nonzero(~pred_img & ~true_img),
        "false_positive": np.count_nonzero(pred_img & ~true_img),
        "false_negative": np.count_nonzero(~pred_img & true_img),
        "precision": mdp.precision(pred_img_bin, true_img_bin),
        "recall": recall,
        "specificity": mdp.specificity(pred_img_bin, true_img_bin),
        "sensitivity": recall,
        "true_positve_rate": recall,
        "true_negative_rate": mdp.true_negative_rate(pred_img_bin, true_img_bin)
    } 

def calc_dice_score(pred_img, true_img, segment):
    # metric in medical image segmentation to evaluate similarity or overlap between segmented images
    # convert images to binary
    true_img_bin = _convert_to_binary(true_img, segment)
    pred_img_bin = _convert_to_binary(pred_img, segment)

    # perform element wise multiplication to get intersection
    intersection = np.sum(pred_img_bin * true_img_bin)
    dice_score = (2.0 * intersection) / (np.sum(true_img_bin) + np.sum(pred_img_bin))
    return dice_score
# endregion

# region Boundary-based metrics
def calc_hausdorff_95(pred_img, true_img, segment):
    return _exec_hausdorff_95(pred_img, true_img, segment, plot=False)

def plot_hausdorff_95(pred_img, true_img, segment):
    return _exec_hausdorff_95(pred_img, true_img, segment, plot=True)

def _calc_surface_distances(pred_img_bin, true_img_bin, voxelspacing=None, connectivity=1):
  """
  The distances between the surface voxel of binary objects in result and their
  nearest partner surface voxel of a binary object in reference.
  
  Source Code: https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L357
  """
  if voxelspacing is not None:
      # ensures that the sequence has the same length as the number of dimensions in the result array
      # not needed for now
      voxelspacing = _ni_support._normalize_sequence(voxelspacing, pred_img_bin.ndim)
      voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
      if not voxelspacing.flags.contiguous:
          voxelspacing = voxelspacing.copy()
          
  # this will create 3x3 binary array with False values at the center
  # and its surrounding structures. diagonals are not considered if connectivity = 1
  footprint = generate_binary_structure(pred_img_bin.ndim, connectivity)

  # extract only 1-pixel border line of objects
  # applies binary erosion to the image
  pred_img_border = pred_img_bin ^ binary_erosion(pred_img_bin, structure=footprint, iterations=1)
  true_img_border = true_img_bin ^ binary_erosion(true_img_bin, structure=footprint, iterations=1)

  # calculate distances from each voxel to the closest border 
  dist = distance_transform_edt(~true_img_border, sampling=voxelspacing)
  surface_dist = np.where(pred_img_border, dist, None)
  return surface_dist

def _exec_hausdorff_95(pred_img, true_img, segment, plot=False):
  # convert image to binary
  true_img_bin = _convert_to_binary(true_img, segment)
  pred_img_bin = _convert_to_binary(pred_img, segment)

  if np.sum(true_img_bin) + np.sum(pred_img_bin) == 0:
    raise Exception("Images do not have common segments")

  connectivity = pred_img_bin.ndim * 2
  surface_dist_pred = _calc_surface_distances(pred_img_bin, true_img_bin, connectivity=connectivity)
  surface_dist_true = _calc_surface_distances(true_img_bin, pred_img_bin, connectivity=connectivity)
  surface_dist_pred_no_null = surface_dist_pred[surface_dist_pred != None]
  surface_dist_true_no_null = surface_dist_true[surface_dist_true != None]
  hd95_val = np.percentile(np.hstack((surface_dist_pred_no_null, surface_dist_true_no_null)), 95)
  
  if plot:
    return hd95_val, surface_dist_pred, surface_dist_true
  else:
    return hd95_val

# endregion

# region Helper Functions
def _convert_to_binary(img, segment):
    return img == segment    

# endregion