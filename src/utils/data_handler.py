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

# All constants for destination directories
# this will be either in google storage or runtime depending on flag
DATA_DIR = "content/data"

# this will always be in google storage
TRAIN_DIR = "training"

SEGMENTS = {
    0: "ELSE",
    1: "NCR",
    2: "ED",
    4: "ET"
}

SEGMENT_COLORS = {
    0: "gray",
    1: "red",
    2: "green",
    4: "yellow"
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
    
class DataHandler:
    """
        Handles all functions related to MRI and training data management such as saving, downloading and reading
        1. Includes reading MRI data functions
        2. Includes reading and storing generic data
        
        Sample Usage:
        1. Instantiate the class. Set use_cloud to False to use runtime instead of google storage. Note that the
        folder structure is consistent in google storage and in runtime.

            data_handler = DataHandler(use_cloud=False)
    
    
        2. Reading MRI images:
            flair_img = data_handler.load_mri(subj_id=subj_id,
                                    mri_type=MriType.STRUCT_SCAN,
                                    struct_scan=StructuralScan.FLAIR)
            
            auto_segmented_img = data_handler.load_mri(subj_id=subj_id,
                                mri_type=MriType.AUTO_SEGMENTED)
                                
            annotated_img = data_handler.load_mri(subj_id=subj_id,
                                mri_type=MriType.ANNOTATED, dtype='uint8')                    
        
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
                 img_dir: str = None, return_nifti: bool = False, dtype: str = None):
        """
        Read MRI data from the specified subject file.

        Args:
            subj_file (str): The path or filename of the MRI subject file (eg. UPENN-GBM-00312_1)
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            struct_scan (Struct_Scan): The type of structural scan to load (e.g T1, T2, T1GD, FLAIR). Strictly use ENUM class for valid structural scans
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
       
        # overwrite constructed file if img_dir is supplied
        if img_dir is not None:
            mri_file_path = img_dir
             
      
        if self.use_cloud:    
            # google storage way
            # create temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(mri_file_path)
            
            self.google_client.download_blob_as_file(mri_file_path, destination_path)
            nifti = nib.load(destination_path)

        else:
            # one drive way
            self.download_from_onedrive(mri_type=mri_type)
            nifti = nib.load(mri_file_path)

      
        # return uint if file is segmentation not specified
        if struct_scan is None and dtype is None:
            dtype="uint"

        # get array data
        data = None
        if dtype is not None:
            data = (nifti.get_fdata()).astype(dtype)
        else:
            data = nifti.get_fdata()
        
        # delete temporary file
        if self.use_cloud:
            os.remove(destination_path)
        
        if return_nifti:
            return data, nifti
        else:
            return data
    
    def load_to_temp_file(self, file_name, train_dir_prefix = None, use_cloud=True):
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)
        
        if self.use_cloud or use_cloud:   
            # google storage way
            # create temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(file_name)
            self.google_client.download_blob_as_file(source_path, destination_path)
            return destination_path
        else:
            # no need to load if from local folder
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
                
    def list_dir(self, sort: bool=True, dir_prefix: str = "", absolute_dir: str = None):
        
        # Look within the TRAIN DIR by default
        dir = os.path.join(self.train_dir, dir_prefix)
        
        if absolute_dir is not None:
            dir = absolute_dir
        
        all_files = []
        if self.use_cloud:
            all_files = self.google_client.list_blob_in_dir(dir)
        else:
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
        
    def get_mri_subj_id(self, file_name):
        # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
        # subj_id: 00006
        if file_name.strip().split('.')[-1] == 'gz':
            return file_name.strip().split('_')[0].split('-')[-1]
        return None
    
    def get_mri_subj(self, file_name):
        # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
        # result: UPENN-GBM-00006
        if file_name.strip().split('.')[-1] == 'gz':
            return file_name.strip().split('_')[0]
        return None
        
    def get_mri_slice_file_name(self, file_name):
        # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
        # result: UPENN-GBM-00006_1.nii.gz
        if file_name.strip().split('.')[-1] == 'gz':
            return file_name.strip().split('_')[0] + \
                        "_" + file_name.strip().split('_')[3]
        return None
    
    def get_mri_file_no(self, file_name):
        # file_name: UPENN-GBM-00006_11_FLAIR_1.nii.gz
        # file_no: 11
        if file_name.strip().split('.')[-1] == 'gz':
            return int(file_name.strip().split('.')[0].split('_')[-1])
        return None
    
    def get_largest_tumor_slice_idx(self, img_data, sum=False):
        non_zero_x = np.count_nonzero(img_data, axis=0)
        if sum is True:
            non_zero_x = np.sum(img_data, axis=0)
        total_y = np.sum(non_zero_x, axis=0 )
        slice_idx = np.argmax(total_y)
        return slice_idx, total_y[slice_idx]
    
    def download_from_onedrive(self, mri_type: MriType):
        
        directory = self._get_mri_dir(mri_type=mri_type)
        filename = directory + ".zip"
        
        if not os.path.exists(filename):
            # Note: For some MRI folder structures, unzipping results to a subdirectory called /data
            # specify unzip path parameter to control where the files go and bypass extra subdirectories as needed
            # eg: data/images_annot_reduced instead of data/data/images_annot_reduced
            
            # TODO: force unizp path for now to match google storage: /content/data/2D_slices_reduced_norm/data/
            dir_2d_slices = os.path.join(self.data_dir, "2D_slices_reduced_norm")
            dir_annot_reduced = os.path.dirname(self.data_dir)
            
            if mri_type == MriType.STRUCT_SCAN:
                download(IMAGES_STRUCTURAL_URL, filename=filename)
            
            elif mri_type == MriType.AUTO_SEGMENTED: 
                download(IMAGES_AUTO_SEGM_URL, filename=filename)
                
            elif mri_type == MriType.ANNOTATED:
                download(IMAGES_ANNOT_URL, filename=filename)
                
            elif mri_type == MriType.STRUCT_SCAN_REDUCED:
                download(IMAGES_ANNOT_REDUCED_URL, filename=filename, unzip_path=dir_annot_reduced)
                
            elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
                download(IMAGES_AUTO_SEGM_REDUCED_URL, filename=filename)
                
            elif mri_type == MriType.ANNOTATED_REDUCED:
                download(IMAGES_ANNOT_REDUCED_URL, filename=filename, unzip_path=dir_annot_reduced)
                
            elif mri_type == MriType.TRAIN_2D:
                download(IMAGES_TRAIN_2D_URL, filename=filename, unzip_path=dir_2d_slices)

            elif mri_type == MriType.VAL_2D:
                download(IMAGES_VAL_2D_URL, filename=filename, unzip_path=dir_2d_slices)

            elif mri_type == MriType.TRAIN_2D_CROSS:
                download(IMAGES_TRAIN_2D_CROSS_URL, filename=filename, unzip_path=dir_2d_slices)
            
            elif mri_type == MriType.VAL_2D_CROSS: 
                download(IMAGES_VAL_2D_CROSS_URL, filename=filename, unzip_path=dir_2d_slices)
    
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
        
        if mri_type == MriType.STRUCT_SCAN:
            file_name = f"{file_name}_{struct_scan}"
            file_dir = os.path.join(file_dir, subj_id)
        
        elif mri_type == MriType.AUTO_SEGMENTED: 
            file_name = f"{file_name}_automated_approx_segm"
            
        elif mri_type == MriType.ANNOTATED:
            file_name = f"{file_name}_segm"
        
        elif mri_type == MriType.STRUCT_SCAN_REDUCED:
            file_name = f"{file_name}_{struct_scan}_cut"
            
        elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
            file_name = f"{file_name}_automated_approx_segm_cut"
            
        elif mri_type == MriType.ANNOTATED_REDUCED:
            file_name = f"{file_name}_segm_cut"
            
        elif mri_type == MriType.TRAIN_2D or mri_type == MriType.VAL_2D or MriType.TRAIN_2D_CROSS or MriType.VAL_2D_CROSS:
            if struct_scan is not None:
                file_name = f"{subj_id}_11_{struct_scan}_{file_no}"
            else:
                file_name = f"{subj_id}_11_segm_{file_no}"
            
        file_name = f"{file_name}.nii.gz"
        f_path = os.path.join(file_dir, file_name)
       
        return f_path

    def _get_mri_dir(self, mri_type: MriType):
        dir_2d_slices = os.path.join(self.data_dir, "2D_slices_reduced_norm", "data")
        
        if mri_type == MriType.STRUCT_SCAN:
            return os.path.join(self.data_dir, IMAGES_STRUCTURAL_FNAME)
        
        elif mri_type == MriType.AUTO_SEGMENTED:
            return os.path.join(self.data_dir, IMAGES_AUTO_SEGM_FNAME)

        elif mri_type == MriType.ANNOTATED:
            return os.path.join(self.data_dir, IMAGES_ANNOT_FNAME)
            
        elif mri_type == MriType.STRUCT_SCAN_REDUCED:
            # same as IMAGES_ANNOT_REDUCED_FNAME
            return os.path.join(self.data_dir, IMAGES_ANNOT_REDUCED_FNAME)

        elif mri_type == MriType.AUTO_SEGMENTED_REDUCED:
            return os.path.join(self.data_dir, IMAGES_AUTO_SEGM_REDUCED_FNAME)

        elif mri_type == MriType.ANNOTATED_REDUCED:
            return os.path.join(self.data_dir, IMAGES_ANNOT_REDUCED_FNAME)

        elif mri_type == MriType.TRAIN_2D:
            return os.path.join(dir_2d_slices, IMAGES_TRAIN_2D_FNAME)

        elif mri_type == MriType.VAL_2D:
            return os.path.join(dir_2d_slices, IMAGES_VAL_2D_FNAME)

        elif mri_type == MriType.TRAIN_2D_CROSS:
            return os.path.join(dir_2d_slices, IMAGES_TRAIN_2D_CROSS_FNAME)
        
        elif mri_type == MriType.VAL_2D_CROSS:
            return os.path.join(dir_2d_slices, IMAGES_VAL_2D_CROSS_FNAME)
        
    def _get_blob_extension(self, file_path):
        suffix = os.path.splitext(file_path)
        if file_path.endswith(".nii.gz"):
            return ".nii.gz"
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
