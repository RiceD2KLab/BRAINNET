import os
import tempfile
import time
from enum import Enum
from io import BytesIO
from typing import Union

import nibabel as nib
import numpy as np
from google.cloud import storage as gcs
from onedrivedownloader import download
    
# All constants from source directories
# Define One Drive urls and file names. files are similar in google storage but unzipped
IMAGES_STRUCTURAL_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ER8oOEAm1ANGlK4sodUPdX0B6_7IxmbRoneyo-RXI2HYOg"
IMAGES_STRUCTURAL_FNAME = "images_structural"

IMAGES_ANNOT_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EbmrLEe1ZgpNkaujtMtlDIEB9rQ0Zj82dOWIttA8sD5lSg"
IMAGES_ANNOT_FNAME = "images_segm" 

IMAGES_ANNOT_REDUCED_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EfqrvokObOJEhovsqLl_sscBgplo836OUdN5kJzAYqPZyg"
IMAGES_ANNOT_REDUCED_FNAME = "images_annot_reduced"

IMAGES_AUTO_SEGM_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EToY-Cli4vxMqYwHx_NZ4JsBi1Lo8tOskj9zb4_AZmDfcg"
IMAGES_AUTO_SEGM_FNAME = "automated_segm"

IMAGES_AUTO_SEGM_REDUCED_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EXwqKvC8QpBBjFQUXzKR1-IBtJeP1hwXUQAoJOneJx4-Hw"
IMAGES_AUTO_SEGM_REDUCED_FNAME = "automated_segm_reduced"

IMAGES_TRAIN_2D_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ETxkBw8cJS9InmWhUUFx0l0BnAES2uSbCaLn0jrMi-HK3Q"
IMAGES_TRAIN_2D_FNAME = "train_2d"
 
IMAGES_VAL_2D_URL   = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EeLd0vd4IuxBn5hPlzO8gz0BT2kDX9Uo9AkNqxR_sjKbMg"
IMAGES_VAL_2D_FNAME = "val_2d" 

IMAGES_TRAIN_2D_CROSS_URL = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EWnXLCwsONBNpzziTciJXt4B68qkcVfZTfSXRcS9gxWWGQ?e=r5xDU4"
IMAGES_TRAIN_2D_CROSS_FNAME = "train_2d_cross"
 
IMAGES_VAL_2D_CROSS_URL   = "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EdKtKyMfy2lJvdGGSSz23yEB34_C4MiEm2BmRE9q451Lgg?e=9dK2NO"
IMAGES_VAL_2D_CROSS_FNAME = "val_2d_cross" 

# Google drive authentication
STORAGE_BUCKET_NAME = "rice_d2k_biocv"
STORAGE_AUTH_FILE = os.path.join("auth", "zinc-citron-387817-2cbfd8289ed2.json")

# All constants for destination directories
DATA_DIR = "content/data"

SEGMENTS = {
    0: "ELSE",
    1: "NCR",
    2: "ED",
    4: "ET"
}

class StructuralScan(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T1GD = "T1GD"
    FLAIR = "FLAIR"

class MriType(Enum):
    STRUCT_SCAN = 1
    AUTO_SEGMENTED = 2
    ANNOTATED = 3
    STRUCT_SCAN_REDUCED = 4
    AUTO_SEGMENTED_REDUCED = 5
    ANNOTATED_REDUCED = 6
    TRAIN_2D = 7
    VAL_2D = 8
    TRAIN_2D_CROSS = 9
    VAL_2D_CROSS = 10
    
class MriImage:
    '''Example Usage:
        mri_img = MriImage(enable_gstorage=True)

        1. To download auto-segmented version of UPENN-GBM-00312_1:
        auto_segm_img = mri_img.read_mri("UPENN-GBM-00312_1", auto_segm=True)
        
        2. To download FLAIR version of UPENN-GBM-00312_1:
        flair_img = mri_img.read_mri("UPENN-GBM-00312_1", struct_scan='FLAIR')
        
        3. To list all struct files:
        mri_img.list_files(self, struct_scan="T1", segm=False, auto_segm=False, reduced=False):
    '''
        
    def __init__(self, enable_gstorage=True):
        """Initialize MriImage with a parameter.
        Args:
            enable_gstorage (int): use runtime storage if set to false
        """
        
        self.enable_gstorage = enable_gstorage
        if self.enable_gstorage:
            # authenticate storage client if specified. if not, runtime will automatically be used
            self.storage_client = gcs.Client.from_service_account_json(STORAGE_AUTH_FILE)
        else:
            # create the directory in the runtime environment
            os.makedirs(DATA_DIR, exist_ok=True)
        
    def load_mri(self, subj_file: str, mri_type: MriType, struct_scan: Union[StructuralScan, None] = None, return_arr: bool = True, dtype: str = None):
        """
        Read MRI data from the specified subject file.

        Args:
            subj_file (str): The path or filename of the MRI subject file (eg. UPENN-GBM-00312_1)
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            struct_scan (Struct_Scan): The type of structural scan to load (e.g T1, T2, T1GD, FLAIR). Strictly use ENUM class for valid structural scans
            return_arr (bool, optional): Flag indicating whether to return the data as an array. Defaults to True.
            dtype (str, optional): The data type of the MRI data e.g. uint16, uint8. If none, default is float.

        Returns:
            The MRI data in the specified format.
        """
        
        # construct the full file path using a helper function
        mri_file_path = self._get_full_path(subj_file=subj_file, mri_type=mri_type, struct_scan=struct_scan)
        temp_file_path = ""
        
        nifti = None
        
        if self.enable_gstorage:    
            # google storage way
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
                temp_file_path = temp_file.name

            # get bucket instance
            bucket = self.storage_client.get_bucket(STORAGE_BUCKET_NAME)
            blob = bucket.blob(mri_file_path)
            blob.download_to_filename(temp_file_path)
            
            nifti = nib.load(temp_file_path)

        else:
            # one drive way
            # download from onedrive if still missing
            self.download_from_onedrive(mri_type=mri_type)
        
            # load file as nifti
            nifti = nib.load(mri_file_path)

        # allow functionality to return raw data instead of array data
        if return_arr==False:
            return nifti
          
        # if segmentation, return uint if not specified
        if mri_type != MriType.STRUCT_SCAN and dtype is None:
            dtype="uint"

        # get array data
        data = None
        if dtype is not None:
            data = nifti.get_fdata().astype(dtype)
        else:
            data = nifti.get_fdata()
        
        # delete temporary file
        if self.enable_gstorage:
            os.remove(temp_file_path)
        
        # return data
        return data
    
    def list_mri_in_folder(self, mri_type: MriType, struct_scan: Union[StructuralScan, None] = None):
        
        all_file_names = []
        directory = self._get_directory(mri_type=mri_type) 
        print("directory", directory)
        
        if self.enable_gstorage:
            # google storage way
            directory = directory.lstrip("/")
            
            bucket = gcs.Bucket(self.storage_client, STORAGE_BUCKET_NAME)

            # list all files in the directory
            all_file_paths = self.storage_client.list_blobs(bucket, prefix=directory)
            all_file_names = [os.path.basename(file.name) for file in all_file_paths]
            
            # probably a mac issue when moving files to google cloud. just delete it from the list of files
            if ".DS_Store" in all_file_names:
                all_file_names.remove(".DS_Store")

            # if structural scan is specified, filter list by the scan type 
            if struct_scan is not None:
                all_file_names = list(filter(lambda item: struct_scan+"_" in item, all_file_names))
                
        else:
            # download if folder is not in runtime storage yet
            self.download_from_onedrive(mri_type=mri_type)
            
            # list dir
            all_file_names = os.listdir(directory)
            
        all_file_names.sort()
        return all_file_names

    def download_from_onedrive(self, mri_type: MriType):
        
        directory = self._get_directory(mri_type=mri_type)
        filename = directory + ".zip"
        
        # Note: For some MRI folder structures, unzipping results to another extra subdirectory called /data
        # specify unzip path to put the files in the same level as the others
        # eg: data/images_annot_reduced instead of data/data/images_annot_reduced
        parent_dir = "/" + os.path.dirname(DATA_DIR)
            
        if not os.path.exists(filename):
            
            if mri_type == MriType.STRUCT_SCAN:
                download(IMAGES_STRUCTURAL_URL, filename=filename)
            
            elif mri_type == MriType.AUTO_SEGMENTED: 
                download(IMAGES_AUTO_SEGM_URL, filename=filename)
                
            elif mri_type == MriType.ANNOTATED:
                download(IMAGES_ANNOT_URL, filename=filename)
                
            elif mri_type == MriType.STRUCT_SCAN_REDUCED:
                download(IMAGES_ANNOT_REDUCED_URL, filename=filename, unzip_path=parent_dir)
                
            elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
                download(IMAGES_AUTO_SEGM_REDUCED_URL, filename=filename)
                
            elif mri_type == MriType.ANNOTATED_REDUCED:
                download(IMAGES_ANNOT_REDUCED_URL, filename=filename, unzip_path=parent_dir)
                
            elif mri_type == MriType.TRAIN_2D:
                download(IMAGES_TRAIN_2D_URL, filename=filename, unzip_path=parent_dir)

            elif mri_type == MriType.VAL_2D:
                download(IMAGES_VAL_2D_URL, filename=filename, unzip_path=parent_dir)

            elif mri_type == MriType.TRAIN_2D_CROSS:
                download(IMAGES_TRAIN_2D_CROSS_URL, filename=filename, unzip_path=parent_dir)
            
            elif mri_type == MriType.VAL_2D_CROSS: 
                download(IMAGES_VAL_2D_CROSS_URL, filename=filename, unzip_path=parent_dir)
            
    def _get_full_path(self, subj_file: str, mri_type: MriType, struct_scan: Union[StructuralScan, None] = None):
        f_name = f"{subj_file}"
        f_dir = self._get_directory(mri_type=mri_type)
        
        # build file name and supply the file path based on current onedrive folder structure
        # the case of structural images is different since each subject has its own folder
        
        if mri_type == MriType.STRUCT_SCAN:
            f_name = f"{f_name}_{struct_scan}"
            f_dir = os.path.join(f_dir, subj_file)
        
        elif mri_type == MriType.AUTO_SEGMENTED: 
            f_name = f"{f_name}_automated_approx_segm"
            
        elif mri_type == MriType.ANNOTATED:
            f_name = f"{f_name}_segm"
        
        elif mri_type == MriType.STRUCT_SCAN_REDUCED:
            f_name = f"{f_name}_{struct_scan}_cut"
            
        elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
            f_name = f"{f_name}_automated_approx_segm_cut"
            
        elif mri_type == MriType.ANNOTATED_REDUCED:
            f_name = f"{f_name}_segm_cut"
            
        elif mri_type == MriType.TRAIN_2D:
            pass

        elif mri_type == MriType.VAL_2D:
            pass

        elif mri_type == MriType.TRAIN_2D_CROSS:
            pass
        
        elif mri_type == MriType.VAL_2D_CROSS: 
            pass
                
        f_name = f"{f_name}.nii.gz"
        f_path = os.path.join(f_dir, f_name)
        
        return f_path

    def _get_directory(self, mri_type: MriType):
        
        dir = ""
        
        if mri_type == MriType.STRUCT_SCAN:
            dir = os.path.join(DATA_DIR, IMAGES_STRUCTURAL_FNAME)
        
        elif mri_type == MriType.AUTO_SEGMENTED:
            dir = os.path.join(DATA_DIR, IMAGES_AUTO_SEGM_FNAME)

        elif mri_type == MriType.ANNOTATED:
            dir = os.path.join(DATA_DIR, IMAGES_ANNOT_FNAME)
            
        elif mri_type == MriType.STRUCT_SCAN_REDUCED:
            # same as IMAGES_ANNOT_REDUCED_FNAME
            dir = os.path.join(DATA_DIR, IMAGES_ANNOT_REDUCED_FNAME)

        elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
            dir = os.path.join(DATA_DIR, IMAGES_AUTO_SEGM_REDUCED_FNAME)

        elif mri_type == MriType.ANNOTATED_REDUCED:
            dir = os.path.join(DATA_DIR, IMAGES_ANNOT_REDUCED_FNAME)

        elif mri_type == MriType.TRAIN_2D:
            dir = os.path.join(DATA_DIR, IMAGES_TRAIN_2D_FNAME)

        elif mri_type == MriType.VAL_2D:
            dir = os.path.join(DATA_DIR, IMAGES_VAL_2D_FNAME)

        elif mri_type == MriType.TRAIN_2D_CROSS:
            dir = os.path.join(DATA_DIR, IMAGES_TRAIN_2D_CROSS_FNAME)
        
        elif mri_type == MriType.VAL_2D_CROSS:
            dir = os.path.join(DATA_DIR, IMAGES_VAL_2D_CROSS_FNAME)
        
        if self.enable_gstorage:
            # google storage needs to see the path as a directory 
            return dir + "/"
        else:
            # runtime drive needs absolute path
            return "/" + dir
    
    def get_largest_tumor_slice_idx(self, img_data, sum=False):
        non_zero_x = np.count_nonzero(img_data, axis=0)
        if sum is True:
            non_zero_x = np.sum(img_data, axis=0)
        total_y = np.sum(non_zero_x, axis=0 )
        slice_idx = np.argmax(total_y)
        return slice_idx, total_y[slice_idx]
    
    def strip_subj_id(self, mri_file_name):
        pass