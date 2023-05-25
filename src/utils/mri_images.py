
import nibabel as nib
import os
from onedrivedownloader import download

DATA_DIR = os.path.join("data")

IMG_SEGM_DIR = os.path.join(DATA_DIR, "images_segm")
IMG_STRUCT_DIR = os.path.join(DATA_DIR, "images_structural")
IMG_REDUCED_DIR = os.path.join(DATA_DIR, "data", "images_annot_reduced")
IMG_AUTO_SEGM_DIR = os.path.join(DATA_DIR, "automated_segm")
IMG_AUTO_SEGM_REDUCED_DIR = os.path.join(DATA_DIR, "automated_segm_reduced")

IMAGES_SEGM_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EbmrLEe1ZgpNkaujtMtlDIEB9rQ0Zj82dOWIttA8sD5lSg"
IMAGES_SEGM_FNAME = "images_segm.zip" 

IMAGES_STRUCTURAL_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ER8oOEAm1ANGlK4sodUPdX0B6_7IxmbRoneyo-RXI2HYOg"
IMAGES_STRUCTURAL_FNAME = "images_structural.zip"

IMAGES_ANNOT_REDUCED_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EfqrvokObOJEhovsqLl_sscBgplo836OUdN5kJzAYqPZyg"
IMAGES_ANNOT_REDUCED_FNAME = "images_annot_reduced.zip"

IMAGES_AUTO_SEGM_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EToY-Cli4vxMqYwHx_NZ4JsBi1Lo8tOskj9zb4_AZmDfcg"
IMAGES_AUTO_SEGM_FNAME = "automated_segm.zip"

IMAGES_AUTO_SEGM_REDUCED_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EXwqKvC8QpBBjFQUXzKR1-IBtJeP1hwXUQAoJOneJx4-Hw"
IMAGES_AUTO_SEGM_REDUCED_FNAME = "automated_segm_reduced.zip"

# file keys
SEGMENTED = "segm"
AUTOSEGMENTED = "auto_segm"
REDUCED = "reduced"
AUTOSEGMENTED_REDUCED = "auto_segm_reduced"
STRUCTURAL = "structural"

STRUCT_SCANS = ["T1", "T2", "T1GD", "FLAIR"]
SEGMENTS = [0, 1, 2, 4]
SEGMENT_NAMES = ["OTHERS", "ED", "NCR/NET", "ET"]

class NiftiImage:
    def __init__(self):
        '''
            Description: Initializes class for handling MRI images
            Args:
                working_dir: google drive directory of the code
                             e.g. '/content/drive/My Drive/Capstone Project/BioCV_Su23/src'
        '''
        os.makedirs(DATA_DIR, exist_ok=True)
        
    def get_img_path(self, subj_file, struct_scan=None, segm=False, auto_segm=False, reduced=False):
        '''
            Description: Constructs the file name and directory for the image file depending on supplied parameters
            Args:
                struct_scan: specify type of scan to be loaded: T1, T2, T1GD, FLAIR
                segm: If true, load the manually annotated image. if struct_scan is not None, this will be overriden 
                    and the structural image will be loaded instead
                auto_segm: If true, load the auto-labeled segmented image
                reduced: If true, load the reduced version of the specified image type. This parameter can be true for any image
                    
            Returns: file name and file path
        '''
        
        # assert that struct_scan should be specified if we are not loading the labelled/segmented images
        if segm == False and auto_segm == False:
            assert struct_scan is not None, f'Specify which struct scan {STRUCT_SCANS} if both segm and auto_segm is False'

        # assert that image cannot be both manual and auto-labelled
        if segm == True and auto_segm == True:
            assert auto_segm == False, 'Can only select either segmented or auto-segmented'

        f_name = f"{subj_file}"
        f_dir = ""

        # build file name and supply the file path based on current onedrive folder structure
        if struct_scan is not None:
            f_name = f"{f_name}_{struct_scan}"
            f_dir = os.path.join(IMG_STRUCT_DIR, subj_file)
        
        if segm:
            f_name = f"{f_name}_segm"
            f_dir = IMG_SEGM_DIR

        if auto_segm:
            f_name = f"{f_name}_automated_approx_segm"
            f_dir = IMG_AUTO_SEGM_DIR

        if reduced:
            f_name = f"{f_name}_cut"
            f_dir = IMG_AUTO_SEGM_REDUCED_DIR if auto_segm==True else IMG_REDUCED_DIR
            
        f_name = f"{f_name}.nii.gz"
        f_path = os.path.join(f_dir, f_name)
        
        return f_name, f_path

    def read(self, subj_file, struct_scan=None, segm=False, auto_segm=False, reduced=False, dtype=None, return_arr=True):
        '''
            Args:
            subj_file: e.g. UPENN-GBM-00001_11
            struct_scan: T1, T2, T1GD, FLAIR
            dtype: e.g. uint. default is float64 if none is supplied
            return_arr: return nibabel object instead of image array if set to False
            
            Description: Loads nifti file depending on supplied parameters
            Returns: file name and loaded image
        '''
        f_name, f_path = self.get_img_path(subj_file, struct_scan=struct_scan, segm=segm, auto_segm=auto_segm, reduced=reduced)
        nifti = nib.load(f_path)
        
        if return_arr==False:
            return f_name, nifti
            
        if auto_segm == True or segm == True:
            dtype="uint"

        if dtype is not None:
            return f_name, nifti.get_fdata().astype(dtype)
        else:
            return f_name, nifti.get_fdata()


    def download(self, struct_scan=False, segm=True, auto_segm=False, reduced=False):

        files = {}
        if struct_scan:
            # 1. baseline pre-operative scans
            download(IMAGES_STRUCTURAL_URL, filename=os.path.join(DATA_DIR, IMAGES_STRUCTURAL_FNAME))
            files[STRUCTURAL] = self._list(IMG_STRUCT_DIR)
        
        if auto_segm:
            # 2. auto-segmented using DeepMedic, DeepSCAN, and nnUNet, combined using STAPLE
            download(IMAGES_AUTO_SEGM_URL, filename=os.path.join(DATA_DIR, IMAGES_AUTO_SEGM_FNAME))
            files[AUTOSEGMENTED] = self._list(IMG_AUTO_SEGM_DIR)

        if segm:
            # 3. manually adjusted/refined labels 
            download(IMAGES_SEGM_URL, filename=os.path.join(DATA_DIR, IMAGES_SEGM_FNAME))
            files[SEGMENTED] = self._list(IMG_SEGM_DIR)

        if reduced and auto_segm:
            # 4. auto-segmented reduced version
            download(IMAGES_AUTO_SEGM_REDUCED_URL, filename=os.path.join(DATA_DIR, IMAGES_AUTO_SEGM_REDUCED_FNAME))
            files[AUTOSEGMENTED_REDUCED] = self._list(IMG_AUTO_SEGM_REDUCED_DIR)
            
        if reduced and (segm or struct_scan):
            # 5. reduced version of structural scans and refined labels
            download(IMAGES_ANNOT_REDUCED_URL, filename=os.path.join(DATA_DIR, IMAGES_ANNOT_REDUCED_FNAME))
            files[REDUCED] = self._list(IMG_REDUCED_DIR)
        
        return files
    
    def _list(self, dir):
        img_files = os.listdir(dir)
        img_files.sort()
        return img_files