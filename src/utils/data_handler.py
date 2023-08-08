import os
import tempfile
import torch
import shutil
import nibabel as nib
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from io import BytesIO
from tqdm.auto import tqdm
from typing import Union, Literal
from onedrivedownloader import download

# custom classes
from utils.google_storage import GStorageClient

# training files fixed to reside in google runtime environment. hence the "/" prefix
DATA_DIR = "/content/data"

# training folder can be stored in workdirectory or cloud
TRAIN_DIR = "training"

class StructuralScan(str, Enum):
    FLAIR = "FLAIR"
    T1 = "T1"
    T1GD = "T1GD"
    T2 = "T2"

class LatentVector(str, Enum):
    LATENT_VECTOR_1 = "latent_vector_1"
    LATENT_VECTOR_2 = "latent_vector_2"
    LATENT_VECTOR_3 = "latent_vector_3"

class SliceDirection(str, Enum):
    DEPTH = "DEPTH"
    CROSS_SIDE = "CROSS_SIDE"
    CROSS_FRONT = "CROSS_FRONT"
    
class MriType(Enum):
    STRUCT_SCAN = 1 # Original MRI Scans: FLAIR, T1, T1GD, T2
    AUTO_SEGMENTED = 2 # Predicted segmentation maps from previous winners of BraTS competition
    ANNOTATED = 3 # Segmentations which are manually annotated by medical experts
    STRUCT_SCAN_REDUCED = 4 # Reduced version of the original MRI Scans: : FLAIR, T1, T1GD, T2
    AUTO_SEGMENTED_REDUCED = 5 # Reduced version of the predicated segmentation maps
    ANNOTATED_REDUCED = 6 # Reduced version of the manually annotated images
    ANNOTATED_REDUCED_NORM = 7 # Normalized versions of AUTO_SEGMENTED_REDUCED and STRUCT_SCAN_REDUCED
    LATENT_SPACE_VECTORS = 8
    LATENT_SPACE_VECTORS_NORM = 9,
    ANNOTATED_REDUCED_NORM_2D = 10

MRI_ONEDRIVE_INFO = {
    MriType.STRUCT_SCAN.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ERmD46rxoZFLjCVwqPPDhaYBe89qKg6Is_TZ8XirWzv7Cw",
        "fname": "images_structural"
    },
    MriType.AUTO_SEGMENTED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EYc4iE_kgOFGnUvw8_3YotIBydkzkvJvaV7Br7VttpMhVg",
        "fname": "automated_segm"
    },
    MriType.ANNOTATED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ERFzCnYX44ZBvxe6ImlnC_0BcGZIkvS1nMebCNovAoe7eA",
        "fname": "images_segm"
    },
    MriType.AUTO_SEGMENTED_REDUCED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/Ec7czJqbusNAlfUVIgq7X4oBxdC75x7nWI-EOAqdtJ_9CQ",
        "fname": "automated_segm_reduced"
    },
    MriType.STRUCT_SCAN_REDUCED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EaWbmywxfmhIqfgmw0TST2ABIckIBmfDQy89EtXtAbz8dQ",
        "fname": "images_annot_reduced",
        "unzip_path": "images_annot_reduced" # For .zip files without a subfolder, this will extract the files into the specified subfolder.
    },
    MriType.ANNOTATED_REDUCED.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EaWbmywxfmhIqfgmw0TST2ABIckIBmfDQy89EtXtAbz8dQ",
        "fname": "images_annot_reduced",
        "unzip_path": "images_annot_reduced" # For .zip files without a subfolder, this will extract the files into the specified subfolder.
    },
    MriType.ANNOTATED_REDUCED_NORM.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EW3iLu-AIvlOgVgMNLhoxHABTCU_aR2T7q0kn4E11uMHlw",
        "fname": "images_annot_reduced_norm",
        "unzip_path": "images_annot_reduced_norm" # For .zip files without a subfolder, this will extract the files into the specified subfolder.
    },
    MriType.LATENT_SPACE_VECTORS.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/ERJkqmA4LGhMlGBY8uCOQYQBigE0_-EW28bLXuf1dXi0hg",
        "fname": "latent_space_vectors",
        "unzip_path": "latent_space_vectors" # For .zip files without a subfolder, this will extract the files into the specified subfolder.
    },
    MriType.LATENT_SPACE_VECTORS_NORM.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EWiAVCh-RV1OqrF-doijvE0BYvKwiiNSV135dU6WNVbMJg",
        "fname": "latent_space_vectors_annot_reduced_norm",
        "unzip_path": "latent_space_vectors_annot_reduced_norm" # For .zip files without a subfolder, this will extract the files into the specified subfolder.
    },
    MriType.ANNOTATED_REDUCED_NORM_2D.name: {
        "url": "https://rice-my.sharepoint.com/:u:/g/personal/hl9_rice_edu/EXHfaGJsMkpIgSej6BfEfs4BJ13HEcaiowZ8Zez85NqlHw",
        "fname": "images_annot_reduced_norm_2d",
        "unzip_path": "images_annot_reduced_norm_2d" # For .zip files without a subfolder, this will extract the files into the specified subfolder.
    },
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

    def __init__(self):
        """Initialize MriImage with a parameter.
        Args:
            use_cloud (int): use runtime storage if set to false
        """

        self.google_client = GStorageClient()
        # this is always in colab runtime
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)


    def load_mri(self, subj_id: str, 
                 mri_type: Union[MriType, None]=None, 
                 file_no: int=None, 
                 struct_scan: Union[StructuralScan, LatentVector, None]=None,
                 dtype: Union[type, np.dtype, None]=None,
                 return_nifti: bool = False,
                 local_path: str=None,
                 dataset_type: Literal["train", "val", "test", None]=None):
        """
        Read MRI data from the specified subject file.

        Args:
            subj_file (str): The path or filename of the MRI subject file (eg. UPENN-GBM-00312)
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            file_no (int): If loading 2d data, this is the slice number
            struct_scan (Struct_Scan): Enum type forloading structural scan (e.g T1, T2, T1GD, FLAIR). Strictly use ENUM class for valid structural scans
            dtype (str, optional): The data type of the MRI data e.g. np.uint8, 'uint8'. If none, default is float.
            return_nifti (bool, optional): Flag indicating whether raw nifit. Defaults to False.
            local_path (str, optional): Local path to the dataset
            dataset_type (str, optional): The dataset folder (e.g train, val, test)
        Returns:
            The MRI data in the specified format.
        """

        # initialize data
        nifti = None

        # download and unzip if the files are from onedrive and do not exist in the runtime yet
        if mri_type is not None:
            self._download_from_onedrive(mri_type=mri_type)
        
            # normalized images have train/test/val
            if mri_type == MriType.ANNOTATED_REDUCED_NORM or \
                mri_type == MriType.ANNOTATED_REDUCED_NORM_2D or \
                    mri_type == MriType.LATENT_SPACE_VECTORS_NORM:
                    assert dataset_type is not None, "Specify if train, val or test"
            
                  
        # construct the full file path of the file
        mri_file_path = self._get_mri_full_path(subj_id=subj_id, 
                                           mri_type=mri_type, 
                                           struct_scan=struct_scan, 
                                           file_no=file_no,
                                           local_path=local_path,
                                           dataset_type=dataset_type)

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

        if return_nifti:
            # return both nifti.get_fdata() and nifti
            return data, nifti
        else:
            # return nifti.get_fdata() only
            return data
        
    def list_mri_in_dir(self, mri_type: [MriType, None]=None, sort: bool=True, local_path: str=None, 
                        return_dir=False, dataset_type: Literal["train", "val", "test", None]=None):

        if mri_type is not None:
            # attempt to download files to runtime first
            self._download_from_onedrive(mri_type=mri_type)
                
            # get associated mri directory 
            mri_dir = self._get_mri_dir(mri_type=mri_type, dataset_type=dataset_type)
        else:
            # attempt to download files to runtime first
            assert local_path is not None
            mri_dir = local_path

        print("mri directory", mri_dir)
        if mri_type == MriType.STRUCT_SCAN \
            or mri_type == MriType.ANNOTATED_REDUCED_NORM \
                or mri_type == MriType.LATENT_SPACE_VECTORS_NORM:
            # force recurrence since the files are grouped within folders
            all_files = self._list_mri_recurse(startpath=mri_dir)
        else:
            all_files = os.listdir(mri_dir)
            
        if sort:
            all_files.sort()
            
        if return_dir:
            return all_files, mri_dir
        else:
            return all_files

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

    def load_to_temp_file(self, file_name, train_dir_prefix = None, use_cloud=True):
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
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

        if use_cloud:
            file_bytes = self.google_client.download_blob_as_bytes(source_path)
            return BytesIO(file_bytes)
        else:
            with open(source_path, 'rb') as f:
                return BytesIO(f.read())

    def load_text_as_list(self, file_name: str, train_dir_prefix = None, use_cloud=True):
        # load from training folder by default
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
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
        if use_cloud:
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
        
        Args:
            file_name (str): The name of the file to be saved.
            source_path (str): The path from which the file will be fetched.
            train_dir_prefix (str, optional): Optional prefix for TRAIN_DIR where file will be saved by default.
            use_cloud (bool, optional): A flag indicating whether the file should also be uploaded to the cloud.
        """

        # build the destination path
        destination_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
            self.google_client.save_from_source_path(source_path, destination_path)
            os.remove(source_path)
        else:
            # move local temp file to correct directory
            shutil.move(source_path, destination_path)

    def save_text(self, file_name, data, train_dir_prefix = None, use_cloud=True):
        """
        Save a file from a source path to a destination directory.
        
        Args:
            file_name (str): The name of the file to be saved.
            source_path (str): The path from which the file will be fetched.
            train_dir_prefix (str, optional): Optional prefix for TRAIN_DIR where file will be saved by default.
            use_cloud (bool, optional): A flag indicating whether the file should also be uploaded to the cloud.
        """

        # build the destination path
        destination_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
            self.google_client.save_text(destination_path, data)
        else:
            with open(destination_path, 'w') as file:
                file.write(data)

    def create_temp_file(self, file_path):
        # this will create a file in the commonly used temporary directory in Python
        # /tmp or %USERPROFILE%\AppData\Local\Temp in windows
        # Lifespan vary depending on os.
        suffix = self._get_blob_extension(file_path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            return temp_file.name

    def _get_mri_dir(self, mri_type: MriType, dataset_type: Literal["train", "val", "test", None]=None):
        odrive_info = MRI_ONEDRIVE_INFO[mri_type.name]
        folder_name = odrive_info["fname"]
        
        if dataset_type is not None:
            return os.path.join(self.data_dir, folder_name, dataset_type)
        else:
            return os.path.join(self.data_dir, folder_name)
        
    def _download_from_onedrive(self, mri_type: MriType):
        directory = self._get_mri_dir(mri_type=mri_type)
        zipname = directory + ".zip"
        if not os.path.exists(directory) or os.path.isdir(directory) is False:
            odrive_info = MRI_ONEDRIVE_INFO[mri_type.name]
            odrive_url = odrive_info['url']
            if 'unzip_path' in odrive_info:
                # This is applicable to .zip files without subfolder
                # Specifying an unzip path will prevent extracting the files onto the data_dir 
                unzip_path = os.path.join(self.data_dir, odrive_info['unzip_path'])
                download(odrive_url, filename=zipname, unzip_path=unzip_path)
            else:
                download(odrive_url, filename=zipname)
                
    def _get_mri_full_path(self, subj_id: str, mri_type: MriType, 
                       struct_scan: Union[StructuralScan, LatentVector, None]=None,
                       file_no: int=None, local_path: str=None, 
                       dataset_type: Literal["train", "val", "test", None]=None):
        
        # this constructs the filepaths for all non-custom dataset/mri  
        file_name = f"{subj_id}"
        if mri_type is not None:
            file_dir = self._get_mri_dir(mri_type=mri_type, dataset_type=dataset_type)
        else:
            # if local path, it is expected a 2d image for training
            # and will have consistent format
            assert local_path is not None
            assert file_no is not None
            
            file_dir = local_path
        
        # build file name and supply the file path based on current onedrive folder structure
        # the case of structural images is different since each subject has its own folder
        # note: for some unknown reason, comparing LSV MriType fails, so use .name attribute

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

        elif mri_type == MriType.ANNOTATED_REDUCED_NORM:
            if struct_scan is not None:
                file_name = f"{file_name}_11_{struct_scan}"
            else:
                file_name = f"{file_name}_11_segm"
        elif mri_type is None and local_path is not None:
            if file_no is not None:
                if struct_scan is not None:
                    file_name = f"{file_name}_11_{struct_scan}_{file_no}"
                else:
                    file_name = f"{file_name}_11_segm_{file_no}"

        file_name = f"{file_name}.nii.gz"
        f_path = os.path.join(file_dir, file_name)

        return f_path
    
    def _get_blob_extension(self, file_path):
        suffix = os.path.splitext(file_path)
        if file_path.endswith(".nii.gz"):
            return ".nii.gz"
        else:
            return suffix[1]

    def _get_train_dir(self, file_name, train_dir_prefix = None, use_cloud = True):
        # default folder is train_dir_prefix
        if train_dir_prefix is None:
            train_dir_prefix = ""

        train_dir = os.path.join(TRAIN_DIR, train_dir_prefix, file_name)
        if use_cloud == False:
            os.makedirs(train_dir, exist_ok=True)

        return train_dir
        
    def _list_mri_recurse(self, startpath):
        file_paths = []
        for _, _, files in os.walk(startpath):
            for file_name in files:
                file_paths.append(file_name)
        return file_paths

    def generate_2d_slices(self, input_dir: str = None, output_dir: str = None, orientation: SliceDirection = None,
                        mri_type: Union[MriType, None] = None):
        """
        Extracts 2d slices from a loaded 3D volumes in the specified orientation

        orientation can be either "DEPTH", "CROSS_FRONT", or "CROSS_SIDE"

        Returns None.
        """
        if mri_type:
            # this is so we won't have to generate our 2d slices many times during training.
            # otherwise we just the function suplied with any input directory
            
            # attempt to download files to runtime first
            self._download_from_onedrive(mri_type=mri_type)
                
            # get associated mri directory 
            output_train_dir = self._get_mri_dir(mri_type=mri_type, dataset_type="train")
            output_val_dir = self._get_mri_dir(mri_type=mri_type, dataset_type="val")
            output_test_dir = self._get_mri_dir(mri_type=mri_type, dataset_type="test")
            
            return [output_train_dir, output_val_dir, output_test_dir]
        
        
        # specify directory paths
        direction = orientation.name.lower()
        output_train_dir = os.path.join(output_dir, "train", direction)
        output_val_dir = os.path.join(output_dir, "val", direction)
        output_test_dir = os.path.join(output_dir, "test", direction)

        # create the directories, assume if train_dir does not exist
        # then neither will val_dir nor test_dir
        if not os.path.exists(output_train_dir):
            os.makedirs(output_train_dir)
            os.makedirs(output_val_dir)
            os.makedirs(output_test_dir)

        assert os.path.exists(input_dir)

        # specify the train/val/test dirs for the loaded 3d mri
        # utils.mri_common.normalize_and_save specifies the subdirs
        # as train/val/test so as long as the volumes were generated
        # using the function, we can expect the following structure:
        # e.g. latent_space_vectors_annot_reduced_norm/train
        # images_annot_reduced_norm/train
        
        loaded_3d_mri_dir_train = os.path.join(input_dir, "train")
        loaded_3d_mri_dir_val = os.path.join(input_dir, "val")
        loaded_3d_mri_dir_test = os.path.join(input_dir, "test")
        print("loaded_3d_mri_dir_train", loaded_3d_mri_dir_train)
        print("loaded_3d_mri_dir_val", loaded_3d_mri_dir_val)
        print("loaded_3d_mri_dir_test", loaded_3d_mri_dir_test)
        
        # iterate over train/val/test dirs and extract 2d slices
        input_dir_list = [loaded_3d_mri_dir_train, loaded_3d_mri_dir_val, loaded_3d_mri_dir_test]
        output_dir_list = [output_train_dir, output_val_dir, output_test_dir]
        print('Extracting 2D slices from 3D volumes (now is a good time to take 5 min coffee break ...)')
        
        for in_subdir, out_subdir in tqdm(zip(input_dir_list, output_dir_list), total=len(input_dir_list)):
            print(f"Working on {in_subdir} --> {out_subdir}")
            
            self._extract_2d_slices(
                orientation=orientation,
                input_dir=in_subdir,
                output_dir=out_subdir
            )
        
        return output_dir_list
        
    def _extract_2d_slices(self, input_dir: str, output_dir: str, orientation: SliceDirection):
        """
        helper function to generate_2d_slices
        """
        # get a listing of files in the input directory
        dir_list = os.listdir(input_dir)
        
        # call _process_volume to create slices in parallel for each file
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(self._process_volume, 
                                dir_list,
                                [input_dir]*len(dir_list), 
                                [output_dir]*len(dir_list), 
                                [orientation]*len(dir_list)),
                        total=len(dir_list)))
            
    def _process_volume(self, infile, input_dir, output_dir, orientation):
        """
        helper function called to generate 2d slices for 1 volume/1 patient
        """
        # load the volume
        nifti = nib.load(os.path.join(input_dir, infile))
        # get the affine transformation matrix
        affine = nifti.affine
        # get the header
        header = nifti.header
        # get the dimensions
        n_height, n_width, n_depth = tuple(nifti.header["dim"][1:4])

        # determine expected # of slices and execute process in parallel
        if orientation == SliceDirection.DEPTH:
            # idx_range = range(n_depth)
            for idx in range(n_depth):
                sliced_data = nifti.get_fdata()[:, :, idx]
                self._process_slice(idx, sliced_data, affine, header, infile, output_dir)
                
        elif orientation == SliceDirection.CROSS_SIDE:
            for idx in range(n_height):
                sliced_data = nifti.get_fdata()[idx, :, :]
                self._process_slice(idx, sliced_data, affine, header, infile, output_dir)
        elif orientation == SliceDirection.CROSS_FRONT:
            for idx in range(n_width):
                sliced_data = nifti.get_fdata()[:, idx, :]
                self._process_slice(idx, sliced_data, affine, header, infile, output_dir)


    def _process_slice(self, idx, sliced_data, affine, header, infile, output_dir):
        """
        helper function to save the a 2d slice
        """
        # convert numpy array to Nifti1Image format
        sliced_nifti = nib.Nifti1Image(sliced_data, affine, header)
        
        # save image
        save_fn = f"{infile.split('.nii.gz')[0]}_{idx}.nii.gz"
        nib.save(sliced_nifti, os.path.join(output_dir, save_fn))