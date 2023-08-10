import os
import tempfile
import torch
import shutil
import nibabel as nib
import numpy as np

from enum import Enum
from io import BytesIO
from onedrivedownloader import download
from typing import Union, Literal

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
    """

    def __init__(self):
        """Initialize DataHandler class """

        # Intializes Google Storage for saving training models and metrics
        self.google_client = GStorageClient()
        
        # This is set to reside inside the Google Colab runtime
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)


    def load_mri(self, subj_id: str,
                 mri_type: Union[MriType, None]=None,
                 file_no: int=None,
                 struct_scan: Union[StructuralScan, LatentVector, None]=None,
                 dtype: Union[type, np.dtype, None]=None,
                 return_nifti: bool = False,
                 local_path: str=None,
                 dataset_type: Literal["train", "val", "test", None]=None,
                 orientation: Literal["depth", "cross_front", "cross_side", None]=None):
        """
        Read MRI data from the specified subject file.

        Args:
            subj_id (str): The unique subject identifier (eg. UPENN-GBM-00312)
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            file_no (int): If loading 2d data, this is the slice number
            struct_scan (Struct_Scan): Enum type forloading structural scan (e.g T1, T2, T1GD, FLAIR). Strictly use ENUM class for valid structural scans
            dtype (str, optional): The data type of the MRI data e.g. np.uint8, 'uint8'. If none, default is float.
            return_nifti (bool, optional): Flag indicating whether raw nifit. Defaults to False.
            local_path (str, optional): Local path to the dataset
            dataset_type (str, optional): The dataset folder (e.g train, val, test)
            orientation (str, optional): The orientation folder (e.g orientation, cross_front, cross_side)
        Returns:
            The MRI data in the specified format.
        """

        # initialize data
        nifti = None

        # download and unzip if the files are from onedrive and do not exist in the runtime yet
        if mri_type is not None:
            self._download_from_onedrive(mri_type=mri_type)

            # normalized images have train/test/val
            if mri_type == MriType.ANNOTATED_REDUCED_NORM or mri_type == MriType.LATENT_SPACE_VECTORS_NORM:
                    assert dataset_type is not None, "Specify if train, val or test"
            
            # prepared 2d slices
            if mri_type == MriType.ANNOTATED_REDUCED_NORM_2D:
                assert dataset_type is not None, "Specify if train, val or test"
                assert orientation is not None, "Specify if depth, cross_side or cross_front"

        # construct the full file path of the file
        mri_file_path = self._get_mri_full_path(subj_id=subj_id,
                                           mri_type=mri_type,
                                           struct_scan=struct_scan,
                                           file_no=file_no,
                                           local_path=local_path,
                                           dataset_type=dataset_type,
                                           orientation=orientation)

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

    def list_mri_in_dir(self, mri_type: Union[MriType, None]=None, sort: bool=True, local_path: str=None,
                        return_dir=False, dataset_type: Literal["train", "val", "test", None]=None,
                        orientation: Literal["depth", "cross_front", "cross_side", None]=None):
        """
        Lists the MRI volumes in a specified directory

        Args:
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            sort (bool): Whether to sort the list of file names
            local_path (str, optional): Local path to the dataset
            return_dir (bool): Flag indicating whether to return directory path
            dataset_type (str, optional): The dataset folder (e.g train, val, test)
            orientation (str, optional): The orientation folder (e.g orientation, cross_front, cross_side)
        Returns:
            A list of strings representing MRI file names in specified directory
        """

        if mri_type is not None:
            
            # get associated mri directory
            if mri_type == MriType.ANNOTATED_REDUCED_NORM_2D:        
                assert dataset_type is not None, "Specify if train, val or test"
                assert orientation is not None, "Specify if depth, cross_side or cross_front"


            # attempt to download files to runtime first
            self._download_from_onedrive(mri_type=mri_type)

            mri_dir = self._get_mri_dir(mri_type=mri_type, dataset_type=dataset_type, orientation=orientation)
        else:
            # attempt to download files to runtime first
            assert local_path is not None
            mri_dir = local_path

        print("mri directory", mri_dir)

        if mri_type == MriType.STRUCT_SCAN:
            # force recurrence for files which are grouped within folders
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
        """
        Determines whether a specified directory path exists

        Args:
            train_dir_prefix (str): path to train directory
            use_could (bool): Flag indicating whether to search google storage bucket or local

        Returns:
            A boolean indicating whether path exists
        """
        source_path = self._get_train_dir(file_name="", train_dir_prefix=train_dir_prefix)
        if use_cloud:
            blobs = self.google_client.list_blob_in_dir(source_path)
            return len(blobs)> 1
        else:
            return os.path.exists(source_path)

    def file_exists(self, train_dir_prefix, file_name, use_cloud=True):
        """
        Determines whether a specified file exists

        Args:
            train_dir_prefix (str): path to train directory
            file_name (str): target file name
            use_cloud (bool): Flag indicating whether to search google storage bucket or local

        Returns:
            A boolean indicating whether path exists
        """
        source_path = self._get_train_dir(file_name=file_name, train_dir_prefix=train_dir_prefix)
        if use_cloud:
            return self.google_client.file_exists(source_path)
        else:
            return os.path.exists(source_path)

    def load_to_temp_file(self, file_name, train_dir_prefix = None, use_cloud=True):
        """
        Downloads file from Google Storage Bucket and saves to local storage

        Args:
            file_name (str): target file name
            train_dir_prefix (str, optional): path to train directory
            use_cloud (bool): Flag indicating whether to search google storage bucket or local

        Returns:
            str path to file save location
        """
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
            # create a local temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(file_name)
            
            # download the file into the temp file path
            self.google_client.download_blob_as_file(source_path, destination_path)
            return destination_path
        else:
            # no need to load if from a local folder
            return source_path

    def load_from_stream(self, file_name, train_dir_prefix = None, use_cloud=True):
        """
        Download file as binary

        Args:
            file_name (str): target file name
            train_dir_prefix(str, optional): path to train directory
            use_cloud (bool): Flag indicating whether to search google storage bucket or local
        """
        # load from training folder by default
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
            file_bytes = self.google_client.download_blob_as_bytes(source_path)
            return BytesIO(file_bytes)
        else:
            with open(source_path, 'rb') as f:
                return BytesIO(f.read())

    def load_text_as_list(self, file_name: str, train_dir_prefix = None, use_cloud=True):
        """
        Read the contents of a file and save them in a list

        Args:
            file_name (str): target file name
            train_dir_prefix (str, optional): path to train directory
            use_cloud (bool): Flag indicating whether to search google storage bucket or local
        """
        # load from training folder
        source_path = self._get_train_dir(file_name, train_dir_prefix, use_cloud)

        if use_cloud:
            # create temp file path with same file format as file being downloaded
            destination_path = self.create_temp_file(source_path)

            # download file from cloud and save into the temp file path (destination path)
            self.google_client.download_blob_as_file(source_path, destination_path)
            
            # assign temp file as the source path
            source_path = destination_path

        # read the downloaded file
        lines = []
        with open(source_path, 'r') as file:
            for line in file:
                lines.append(line.strip())
        return lines

    def load_torch(self, file_name, train_dir_prefix, device):
        """
        Loads pytorch data from the cloud

        Args:
            file_name (str): target file name, ex: model.pt
            train_dir_prefix (str): path to train directory

        Returns:
            a torch file
        """
        model_stream = self.load_from_stream(file_name=file_name, train_dir_prefix=train_dir_prefix, use_cloud=True)
        return torch.load(model_stream, map_location=device)

    def save_torch(self, file_name:str, data, train_dir_prefix = None):
        """
        Saves a pytorch data to the cloud

        Args:
            file_name (str): target file name, ex: model.pt
            model (torch.nn.Module): torch model to save
            train_dir_prefix (str, optional): path to train directory

        Returns:
            None
        """
        
        # Training models is fixed to be always uploaded to the cloud
        source_path = self.create_temp_file(file_name)
        torch.save(data, source_path)
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
            # moves local temp file to correct directory
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
        """
        Create a temporary file in /tmp or %USERPROFILE%\AppData\Local\Temp
        Note, lifespan of temporary files varies depending on OS.

        Args:
            file_path (str): path to save file

        Returns:
            str of saved file name
        """
        suffix = self._get_blob_extension(file_path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            return temp_file.name

    def _get_mri_dir(self, mri_type: MriType, 
                     dataset_type: Literal["train", "val", "test", None]=None,
                     orientation: Literal["depth", "cross_front", "cross_side", None]=None):
        """
        Helper function to get folder path for MRI based on MriType and dataset type

        Args:
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            dataset_type (str, optional): The dataset folder (e.g train, val, test)
            orientation (str, optional): The orientation folder (e.g orientation, cross_front, cross_side)
        Returns:
            str to corresponding MRI directory
        """
        odrive_info = MRI_ONEDRIVE_INFO[mri_type.name]
        folder_name = odrive_info["fname"]

        if dataset_type is not None:
            if orientation is not None:
                return os.path.join(self.data_dir, folder_name, dataset_type, orientation)
            return os.path.join(self.data_dir, folder_name, dataset_type)
        else:
            return os.path.join(self.data_dir, folder_name)

    def _download_from_onedrive(self, mri_type: MriType):
        """
        Helper function for downloading files from OneDrive.
        Wrapper for OneDriveDownloader

        Args:
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced)

        Returns:
            None
        """
        # To download, just get the base folder of the zip file in one drive
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
                       dataset_type: Literal["train", "val", "test", None]=None,
                       orientation: Literal["depth", "cross_front", "cross_side", None]=None):
        """
        Helper function for getting full path to a specific MRI file based on MriType

        Args:
            subj_id (str): Unique subject identifier, Ex: `UPENN-GBM-00006`
            mri_type (Mri_Type): The type of MRI data being read (e.g. annotated, autosegmented, reduced). Strictly use ENUM class for valid types
            struct_scan (Struct_Scan): Enum type forloading structural scan (e.g T1, T2, T1GD, FLAIR). Strictly use ENUM class for valid structural scans
            file_no (int): If loading 2d data, this is the slice number
            local_path (str, optional): Local path to the dataset
            dataset_type (str, optional): The dataset folder (e.g train, val, test)
            orientation (str, optional): The orientation folder (e.g orientation, cross_front, cross_side)
        Returns:
            str representing full path to specified file
        """

        # this constructs the filepaths for all raw and pre-trained dataset
        file_name = f"{subj_id}"
        
        # get associated mri directory
        if mri_type is not None:
            file_dir = self._get_mri_dir(mri_type=mri_type, dataset_type=dataset_type, orientation=orientation)
        else:
            # accepted local path is only for 2d slices 
            assert local_path is not None
            assert file_no is not None

            file_dir = local_path

        # build file name and supply the file path based on current onedrive folder structure
        
        if mri_type == MriType.STRUCT_SCAN:
            file_name = f"{file_name}_11_{struct_scan}"
            
            # the case of structural images is different since each subject has its own folder
            # specify its own file directory
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
        """
        Helper function to get file extension. Ex: MRI files end in `.nii.gz`

        Args:
            file_path (str): path to target file

        Returns:
            a str representing file extension type.
        """
        suffix = os.path.splitext(file_path)
        if file_path.endswith(".nii.gz"):
            return ".nii.gz"
        else:
            return suffix[1]

    def _get_train_dir(self, file_name, train_dir_prefix = None, use_cloud = True):
        """
        Helper function to get train directory path

        Args:
            file_name (str): target file name
            train_dir_prefix (str, optional): path to train directory
            use_cloud (bool): Flag indicating whether to use google storage bucket or local

        Returns:
            A str representing path to train directory
        """
        # default folder is train_dir_prefix
        if train_dir_prefix is None:
            train_dir_prefix = ""

        train_dir = os.path.join(TRAIN_DIR, train_dir_prefix, file_name)
        if use_cloud == False:
            os.makedirs(train_dir, exist_ok=True)

        return train_dir

    def _list_mri_recurse(self, startpath):
        """
        Helper function to list MRIs in a series of subdirectories
        from a starting parent directory.

        Args:
            startpath (str): parent directory to start search

        Returns:
            A list of str containing file names
        """
        file_paths = []
        for _, _, files in os.walk(startpath):
            for file_name in files:
                file_paths.append(file_name)
        return file_paths
