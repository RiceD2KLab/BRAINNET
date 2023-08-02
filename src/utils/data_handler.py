import os
import tempfile
import torch
from enum import Enum
from io import BytesIO
from typing import Union
import shutil

import nibabel as nib
import numpy as np

import importlib
import utils.google_storage as gs
importlib.reload(gs)

from utils.google_storage import GStorageClient
from onedrivedownloader import download
    
# All constants for destination directories
# this will be either in google storage or runtime depending on flag
DATA_DIR = "content/data"

# this will always be in google storage
TRAIN_DIR = "training"

class StructuralScan(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T1GD = "T1GD"
    FLAIR = "FLAIR"

class MriType(Enum):
    STRUCT_SCAN = 1 # Original MRI Scans: FLAIR, T1, T1GD, T2
    AUTO_SEGMENTED = 2
    ANNOTATED = 3 # Manually annotated by medical experts
    STRUCT_SCAN_REDUCED = 4
    AUTO_SEGMENTED_REDUCED = 5
    ANNOTATED_REDUCED = 6 # Manually annotated by medical experts - zero reduction
    TRAIN_2D_DEPTH = 7
    VAL_2D_DEPTH = 8
    TEST_2D_DEPTH = 9
    TRAIN_2D_CROSS_SIDE = 10
    VAL_2D_CROSS_SIDE = 11
    TEST_2D_CROSS_SIDE = 12
    TRAIN_2D_CROSS_FRONT = 13
    VAL_2D_CROSS_FRONT = 14
    TEST_2D_CROSS_FRONT = 15
    ANNOTATED_REDUCED_NORM = 16 #  # Manually annotated by medical experts - zero reduction and normalized
    TRAIN_AE_2D_DEPTH = 17
    VAL_AE_2D_DEPTH = 18
    TEST_AE_2D_DEPTH = 19
    TRAIN_AE_2D_CROSS_SIDE = 20
    VAL_AE_2D_CROSS_SIDE = 21
    TEST_AE_2D_CROSS_SIDE = 22
    TRAIN_AE_2D_CROSS_FRONT = 23
    VAL_AE_2D_CROSS_FRONT = 24
    TEST_AE_2D_CROSS_FRONT = 25


MRI_ONEDRIVE_INFO = {
    MriType.STRUCT_SCAN.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ER8oOEAm1ANGlK4sodUPdX0B6_7IxmbRoneyo-RXI2HYOg",
        "fname": "images_structural"
    },
    MriType.AUTO_SEGMENTED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EToY-Cli4vxMqYwHx_NZ4JsBi1Lo8tOskj9zb4_AZmDfcg",
        "fname": "automated_segm"
    },
    MriType.ANNOTATED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EbmrLEe1ZgpNkaujtMtlDIEB9rQ0Zj82dOWIttA8sD5lSg",
        "fname": "images_segm"
    },
    MriType.AUTO_SEGMENTED_REDUCED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EXwqKvC8QpBBjFQUXzKR1-IBtJeP1hwXUQAoJOneJx4-Hw",
        "fname": "automated_segm_reduced"
    },
    MriType.STRUCT_SCAN_REDUCED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EfqrvokObOJEhovsqLl_sscBgplo836OUdN5kJzAYqPZyg",
        "fname": "images_annot_reduced",
        "unzip_path": ""
    },
    MriType.ANNOTATED_REDUCED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EfqrvokObOJEhovsqLl_sscBgplo836OUdN5kJzAYqPZyg",
        "fname": "images_annot_reduced"
    },
    MriType.ANNOTATED_REDUCED_NORM.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EccpxJhE8T5BgDkvbgUr6kIBPG0Nx9dneBeaqPPZ0YlZhw",
        "fname": "images_annot_reduced_norm"
    },
    MriType.TRAIN_2D_DEPTH.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EWo-2IfjoUNImdWqJxc2iywBE1_8q8kVZHIhd9eOFv1wFg",
        "fname": "train_2d",
        "unzip_path": "2D_slices_reduced_norm", # custom unzip path different from how the zip file is structured,
        "has_data_subfolder": True
    },
    MriType.VAL_2D_DEPTH.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ERKhTBiFCUlBpn2L5aG2-CkBMDIJBnLZhzqjkpJeFOIQVQ",
        "fname": "val_2d",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True # sometimes after extracting, the folders still have extra data subfolder. need to consider this
    },
    MriType.TEST_2D_DEPTH.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EeMV2GRSQWdPkQpDmQouC8gBGqGAFahtyiZFCdrkHoIk1w",
        "fname": "test_2d_depth",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True 
    },
    MriType.TRAIN_2D_CROSS_SIDE.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EeYmGZIuupROvu0xpljaJBsBueQcPuMC_sM8nzwdYDcrMg",
        "fname": "train_2d_cross",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True 
    },
    MriType.VAL_2D_CROSS_SIDE.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ERS3vGnRyPpLoiN-PLDUaTQBuGvbv9RaV-Xs-UYNmz6GWA",
        "fname": "val_2d_cross",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True 
    },
    MriType.TEST_2D_CROSS_SIDE.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EcwJqz2sLttFs0Ujni4MjhkBSrxcOzHnOrbzCeiUXBeiTQ",
        "fname": "test_2d_cross_side",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TRAIN_2D_CROSS_FRONT.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EQHcZZBwSw1CgFqvJnz2b7cBno-js9YMS_GcKl9jffaylg",
        "fname": "train_2d_cross_front",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.VAL_2D_CROSS_FRONT.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EZatriq7t2lHkrcc5s11IPoBcm2kDSAtXLuFvRQ1tLUOMA",
        "fname": "val_2d_cross_front",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TEST_2D_CROSS_FRONT.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ETeMow8TKc1BgdsdSBxSLPMBHT-Iq2pZN_OIcsp7HfcQwg",
        "fname": "test_2d_cross_front",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True 
    },
    MriType.TRAIN_AE_2D_DEPTH.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.VAL_AE_2D_DEPTH.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TEST_AE_2D_DEPTH.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TRAIN_AE_2D_CROSS_SIDE.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.VAL_AE_2D_CROSS_SIDE.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TEST_AE_2D_CROSS_SIDE.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TRAIN_AE_2D_CROSS_FRONT.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.VAL_AE_2D_CROSS_FRONT.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    },
    MriType.TEST_AE_2D_CROSS_FRONT.name: {
        "url": "",
        "fname": "",
        "unzip_path": "2D_slices_reduced_norm",
        "has_data_subfolder": True
    }
}
class DataHandler:
    """
        Handles all functions related to MRI and training data management such as saving, downloading and reading
        1. Includes reading MRI data functions
        2. Includes reading and storing generic data
        
        Sample Usage:
        1. Instantiate the class. Set use_cloud to False to use runtime instead of google storage. Note that the
        folder structure is consistent in google storage and in runtime.

            data_handler = DataHandler()
    
    
        2. Reading MRI images:
            flair_img = data_handler.load_mri(subj_id=subj_id,
                                    mri_type=MriType.STRUCT_SCAN,
                                    struct_scan=StructuralScan.FLAIR)
            
            auto_segmented_img = data_handler.load_mri(subj_id=subj_id,
                                mri_type=MriType.AUTO_SEGMENTED)
                                
            annotated_img = data_handler.load_mri(subj_id=subj_id,
                                mri_type=MriType.ANNOTATED, dtype=np.uint8)                    
        
        3. Save Text:
            data_handler.save_text(filename='filename.txt', data='this is my text')
            
        4. Upload local file to cloud
            data_handler.save_from_source_path(file_name="destination_filename.txt", source_path="local_file.txt")
        
        5. Download text as list:
            my_list = data_handler.load_text_as_list(filename="filename.txt")
            
        6. Download file as stream:
            model_steam = data_handler.load_from_stream("model_current.pt")
            model = torch.load(model_steam)
        
        7. List Mri Files
            annot_files = data_handler.list_mri_in_dir(mri_type=MriType.ANNOTATED)
            print("No. of segmented files:", len(annot_files))
            
            Output: 
            No. of segmented files: 147
    """
                
    def __init__(self, use_cloud=False):
        """Initialize MriImage with a parameter.
        Args:
            use_cloud (int): use runtime storage if set to false
        """
        
        self.google_client = GStorageClient()
        self.use_cloud = use_cloud
        self.train_dir = TRAIN_DIR
        
        if self.use_cloud:
            # google storage needs to see the path as a directory explicity
            self.data_dir = DATA_DIR + "/"
        else:
            # runtime drive needs absolute path, hence the prefix
            self.data_dir = "/" + DATA_DIR
            os.makedirs(self.data_dir, exist_ok=True)
            
            # manually create training directory if files are saved locally
            os.makedirs(self.train_dir, exist_ok=True)


    def load_mri(self, subj_id: str, mri_type: MriType, file_no: int = None, struct_scan: Union[StructuralScan, None] = None, 
                 return_nifti: bool = False, dtype: Union[type, np.dtype, None] = None):
        """
        Read MRI data from the specified subject file.

        Args:
            subj_file (str): The path or filename of the MRI subject file (eg. UPENN-GBM-00312)
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            file_no (int): If loading 2d data, this is the slice number
            struct_scan (Struct_Scan): Enum type forloading structural scan (e.g T1, T2, T1GD, FLAIR). Strictly use ENUM class for valid structural scans
            return_nifti (bool, optional): Flag indicating whether raw nifit. Defaults to False.
            dtype (str, optional): The data type of the MRI data e.g. uint16, uint8. If none, default is float.

        Returns:
            The MRI data in the specified format.
        """
         
        # initialize data
        destination_path = None
        nifti = None
        
        # construct the full file path using a helper function
        mri_file_path = self._get_full_path(subj_id=subj_id, mri_type=mri_type, struct_scan=struct_scan, file_no=file_no)
       
        if self.use_cloud:    
            # if source is google storage
            # create a local temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(mri_file_path)
            
            # download the file into that local destination path
            self.google_client.download_blob_as_file(mri_file_path, destination_path)
            nifti = nib.load(destination_path)

        else:
            # if source is one drive
            # download and unzip if the files do not exist in the runtime yet
            self.download_from_onedrive(mri_type=mri_type)
            
            nifti = nib.load(mri_file_path)

        # return uint8 if image being loaded is a segmentation image
        if struct_scan is None and dtype is None:
            dtype=np.uint8
        
        # get array data
        data = nifti.get_fdata()
        
        if dtype is not None:        
            # BUG FIX: decimal bug on segments: [0 1.0000152587890625, 2.000030517578125 3.9999999990686774]
            # if dtype is of type integer, eg. np.uint8, uint8, np.uint32 etc, round the data before converting to an integer
            if np.issubdtype(dtype, np.integer):
                data = np.round(data).astype(dtype)
            else:
                data = data.astype(dtype)
        
        
        # delete temporary destination path after extracting nifti image
        if self.use_cloud:
            os.remove(destination_path)
        
        if return_nifti:
            # return both nifti.get_fdata() and nifti
            return data, nifti
        else:
            # return nifti.get_fdata() only
            return data
    
    def load_to_temp_file(self, file_name, train_dir_prefix = None, use_cloud=True):
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)
        
        if self.use_cloud or use_cloud:   
            # if source is google storage
            # create a local temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(file_name)
            self.google_client.download_blob_as_file(source_path, destination_path)
            return destination_path
        else:
            # no need to load if from a local folder
            return source_path
            
    def load_from_stream(self, file_name, train_dir_prefix = None, use_cloud=True):
        # load from training folder by default
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)
        
        if self.use_cloud or use_cloud:
            file_bytes = self.google_client.download_blob_as_bytes(source_path)
            return BytesIO(file_bytes)
        else:
            with open(source_path, 'rb') as f:
                return BytesIO(f.read())
        
    def load_text_as_list(self, file_name: str, train_dir_prefix = None, use_cloud=True):
        # load from training folder by default
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)
        
        if self.use_cloud or use_cloud:    
            # create temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(source_path)
            
            # download file and save to temp file path created
            self.google_client.download_blob_as_file(source_path, destination_path)
            # assign temp file as the source path
            source_path = destination_path

        # read the downloaded file
        lines = []
        with open(source_path, 'r') as file:
            for line in file:
                lines.append(line.strip())
        return lines
        
    def load_torch_model(self, file_name, train_dir_prefix, device, use_cloud=True):
        if self.use_cloud or use_cloud:
            model_stream = self.load_from_stream(file_name=file_name, train_dir_prefix=train_dir_prefix, use_cloud=True)
            return torch.load(model_stream, map_location=device)
        else:
            return torch.load(model_stream, map_location=device)
    
    def save_torch_model(self, file_name:str, model, train_dir_prefix = None):
        
        source_path = self.create_temp_file(file_name)
        torch.save(model, source_path)
        
        self.save_from_source_path(file_name, train_dir_prefix=train_dir_prefix, 
                                    source_path=source_path, use_cloud=True)

    def save_from_source_path(self, file_name, source_path, train_dir_prefix = None, use_cloud=True):
        
        """
        Save a file from a source path to a destination directory.
        Optionally, the file can be uploaded to the cloud even if self.use_cloud == False

        Args:
            file_name (str): The name of the file to be saved.
            source_path (str): The path from which the file will be fetched.
            train_dir_prefix (str, optional): Optional prefix for TRAIN_DIR where file will be saved by default.
            data_dir_prefix (str, optional): If specified, file will be uploaded to DATA_DIR instead of TRAIN_DIR.
            use_cloud (bool, optional): A flag indicating whether the file should also be uploaded to the cloud.
        """

        # build the destination path
        destination_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)
        
        if self.use_cloud or use_cloud:
            self.google_client.save_from_source_path(source_path, destination_path)
            os.remove(source_path)
        else:
            # move local temp file to correct directory
            shutil.move(source_path, destination_path)
                 
    def save_text(self, file_name, data, train_dir_prefix = None, use_cloud=True):
        """
        Save a file from a source path to a destination directory.
        Optionally, the file can be uploaded to the cloud even if self.use_cloud == False

        Args:
            file_name (str): The name of the file to be saved.
            source_path (str): The path from which the file will be fetched.
            train_dir_prefix (str, optional): Optional prefix for TRAIN_DIR where file will be saved by default.
            use_cloud (bool, optional): A flag indicating whether the file should also be uploaded to the cloud.
        """
        
        # build the destination path
        destination_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)
        
        if self.use_cloud or use_cloud:
            self.google_client.save_text(destination_path, data)
        else:
            with open(destination_path, 'w') as file:
                file.write(data)
      
    def list_dir(self, train_dir_prefix: str = "", absolute_dir: str = None, sort: bool=True, use_cloud=False):
        """
        List files in a directory
        
        Args:
            sort (str): The name of the file to be saved.
            source_path (str): The path from which the file will be fetched.
            train_dir_prefix (str, optional): Optional subfolder within the train directory
            absolute_dir (str, optional): Optional prefix for TRAIN_DIR which will serve as base directory
            use_cloud (bool, optional): A flag indicating whether the file should also be uploaded to the cloud.
        """
        
        # Look within the train dir by default
        dir = os.path.join(self.train_dir, train_dir_prefix)
        if absolute_dir is not None:
            dir = absolute_dir
        
        all_files = []
        if self.use_cloud or use_cloud:
            # list files from google cloud
            all_files = self.google_client.list_blob_in_dir(dir)
        else:
            # list files from local directory
            all_files = os.listdir(dir)
        
        if sort:
            all_files.sort()
        return all_files
        
    def list_mri_in_dir(self, mri_type: MriType, sort: bool=True):
        
        dir = self._get_mri_dir(mri_type=mri_type)
        if self.use_cloud == False:
            # attempt to download files to runtime first
            self.download_from_onedrive(mri_type=mri_type)
        
        return self.list_dir(absolute_dir=dir, sort=sort)
    
    def dir_exists(self, train_dir_prefix, use_cloud=True):
        source_path = self._get_train_dir(file_name="", train_dir_prefix=train_dir_prefix)
        if use_cloud:
            blobs = self.google_client.list_blob_in_dir(source_path)
            return len(blobs)> 1
        else:
            return os.path.exists(source_path)
        
    def file_exists(self, train_dir_prefix, file_name, use_cloud=True):
        source_path = self._get_train_dir(file_name=file_name, train_dir_prefix=train_dir_prefix)
        if use_cloud:
            return self.google_client.file_exists(source_path)
        else:
            return os.path.exists(source_path)
        
    def download_from_onedrive(self, mri_type: MriType):
        
        directory = self._get_mri_dir(mri_type=mri_type)
        filename = directory + ".zip"
        
        if not os.path.exists(filename):
            # Note: For some MRI folder structures, unzipping results to a subdirectory called /data
            # specify unzip_path parameter to control where the files go and bypass extra subdirectories as needed
            # eg: data/images_annot_reduced instead of data/data/images_annot_reduced
            
            odrive_info = MRI_ONEDRIVE_INFO[mri_type.name]
            odrive_url = odrive_info['url']
            
            has_parent_data_dir = [
                MriType.STRUCT_SCAN_REDUCED,
                MriType.ANNOTATED_REDUCED,
                MriType.ANNOTATED_REDUCED_NORM
            ] 
            if mri_type in has_parent_data_dir:
                # bypass to use data/images_annot_reduced as unzip path instead of data/data/images_annot_reduced
                dir_annot_reduced = os.path.dirname(self.data_dir)
                download(odrive_url, filename=filename, unzip_path=dir_annot_reduced)
            elif "unzip_path" in odrive_info:
                # if unzip path is specified
                odrive_unzip_path = odrive_info['unzip_path']
                unzip_path = os.path.join(self.data_dir, odrive_unzip_path)
                download(odrive_url, filename=filename, unzip_path=unzip_path)
            else:
                download(odrive_url, filename=filename)

    
    def create_temp_file(self, file_path):
        # this will create a file in the commonly used temporary directory in Python 
        # /tmp or %USERPROFILE%\AppData\Local\Temp in windows
        # Lifespan vary depending on os.
        suffix = self._get_blob_extension(file_path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            return temp_file.name
                
    def _get_full_path(self, subj_id: str, mri_type: MriType, file_no: int = None, struct_scan: Union[StructuralScan, None] = None):
        file_name = f"{subj_id}"
        file_dir = self._get_mri_dir(mri_type=mri_type)
        
        # build file name and supply the file path based on current onedrive folder structure
        # the case of structural images is different since each subject has its own folder
        modelling_dataset = [
            MriType.TRAIN_2D_DEPTH,
            MriType.TRAIN_2D_CROSS_SIDE,
            MriType.TRAIN_2D_CROSS_FRONT,
            MriType.VAL_2D_DEPTH,
            MriType.VAL_2D_CROSS_SIDE,
            MriType.VAL_2D_CROSS_FRONT,
            MriType.TEST_2D_DEPTH,
            MriType.TEST_2D_CROSS_SIDE,
            MriType.TEST_2D_CROSS_FRONT
        ]
        
        if mri_type == MriType.STRUCT_SCAN:
            file_name = f"{file_name}_11_{struct_scan}"
            file_dir = os.path.join(file_dir, subj_id) + "_11"
        
        elif mri_type == MriType.AUTO_SEGMENTED: 
            file_name = f"{file_name}_11_automated_approx_segm"
            
        elif mri_type == MriType.ANNOTATED:
            file_name = f"{file_name}_11_segm"
        
        elif mri_type == MriType.STRUCT_SCAN_REDUCED:
            file_name = f"{file_name}_11_{struct_scan}_cut"
            
        elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
            file_name = f"{file_name}_11_automated_approx_segm_cut"
            
        elif mri_type == MriType.ANNOTATED_REDUCED:
            file_name = f"{file_name}_11_segm_cut"
            
        elif mri_type in modelling_dataset:
            if struct_scan is not None:
                file_name = f"{file_name}_11_{struct_scan}_{file_no}"
            else:
                file_name = f"{file_name}_11_segm_{file_no}"
                
        elif mri_type == MriType.ANNOTATED_REDUCED_NORM:
            if struct_scan is not None:
                file_name = f"{file_name}_11_{struct_scan}"
            else:
                file_name = f"{file_name}_11_segm"
            
        file_name = f"{file_name}.nii.gz"
        f_path = os.path.join(file_dir, file_name)
       
        return f_path

    def _get_mri_dir(self, mri_type: MriType):
        odrive_info = MRI_ONEDRIVE_INFO[mri_type.name]
        fname = odrive_info["fname"]
        data_dir = ""
    
        if "unzip_path" in odrive_info:    
            unzip_path = odrive_info["unzip_path"]
            data_dir = unzip_path
        
        if "has_data_subfolder" in odrive_info:
            data_dir = os.path.join(data_dir, "data")
    
        return os.path.join(self.data_dir, data_dir, fname)
        
    def _get_blob_extension(self, file_path):
        suffix = os.path.splitext(file_path)
        if file_path.endswith(".nii.gz"):
            return ".nii.gz"
        elif file_path.endswith(".tii.gz"):
            return ".tii.gz"
        else: 
            return suffix[1]
        
    def _get_train_dir(self, file_name, train_dir_prefix = None, use_cloud = True):
        # default folder is training/train_dir_prefix
        if train_dir_prefix is None:
            train_dir_prefix = ""
        
        train_dir = os.path.join(self.train_dir, train_dir_prefix, file_name)
        
        if self.use_cloud == False and use_cloud == False:
            os.makedirs(train_dir, exist_ok=True)

        return train_dir
