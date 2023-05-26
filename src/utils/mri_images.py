
import nibabel as nib
import os
from google.cloud import storage
import time

DATA_DIR = "data"
TEMP_DIR = "temp_files"

IMG_SEGM_DIR = os.path.join(DATA_DIR, "images_segm")
IMG_STRUCT_DIR = os.path.join(DATA_DIR, "images_structural")
IMG_REDUCED_DIR = os.path.join(DATA_DIR, "images_annot_reduced")
IMG_AUTO_SEGM_DIR = os.path.join(DATA_DIR, "automated_segm")
IMG_AUTO_SEGM_REDUCED_DIR = os.path.join(DATA_DIR, "automated_segm_reduced")

STRUCT_SCANS = ["T1", "T2", "T1GD", "FLAIR"]
SEGMENTS = [0, 1, 2, 4]
SEGMENT_NAMES = ["OTHERS", "ED", "NCR/NET", "ET"]

STORAGE_BUCKET_NAME = "rice_d2k_biocv"
STORAGE_AUTH_FILE = os.path.join("keys", "zinc-citron-387817-2cbfd8289ed2.json")

class MriImage:
    def __init__(self):
        '''
            Description: Initializes class for handling MRI images
            Args:
                working_dir: google drive directory of the code
                             e.g. '/content/drive/My Drive/Capstone Project/BioCV_Su23/src'
        '''
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # authenticate client. key file should not be uploaded to git
        # will just send the .json file via discord if needed
        self.storage_client = storage.Client.from_service_account_json(STORAGE_AUTH_FILE)
        
    # download from google cloud
    def read(self, subj_file, struct_scan=None, segm=False, auto_segm=False, reduced=False, return_arr=True, 
        dtype = None):
        f_name, blob_name = self._get_img_path(subj_file, struct_scan=struct_scan, segm=segm, auto_segm=auto_segm, 
                                               reduced=reduced)
        # get bucket instance
        bucket = self.storage_client.get_bucket(STORAGE_BUCKET_NAME)
        
        # download blob
        blob = bucket.blob(blob_name)
        
        # extract filename from blob directory
        file_name = os.path.basename(blob_name)
        
        # save blob to local temporary file
        # add timestamp to prevent overwriting files with same name
        now = int(time.time())
        temp_file = os.path.join(TEMP_DIR, f"{now}-{file_name}")
        blob.download_to_filename(temp_file)
        
        # load temp file as nifti
        nifti = nib.load(temp_file)

        if return_arr==False:
            return f_name, nifti
          
        if auto_segm == True or segm == True:
            dtype="uint"

        # get data
        data = None
        if dtype is not None:
            data = nifti.get_fdata().astype(dtype)
        else:
            data = nifti.get_fdata()
        
        # delete temporary file
        os.remove(temp_file)
            
        return f_name, data
    
    def list_blobs_in_folder(self, struct_scan=None, segm=False, auto_segm=False, reduced=False):
        """List the blobs (files) in a folder within a Google Cloud Storage bucket"""
        
        # Prefix the folder name with a slash to list blobs within the folder
        bucket = storage.Bucket(self.storage_client, STORAGE_BUCKET_NAME)

        folder_prefix = self._get_folder_path(struct_scan, segm, auto_segm, reduced) + '/'
        all_blobs = self.storage_client.list_blobs(bucket, prefix=folder_prefix)
        all_blob_names = [os.path.basename(blob.name) for blob in all_blobs]
        all_blob_names.sort()
        return all_blob_names

    def _get_img_path(self, subj_file, struct_scan=None, segm=False, auto_segm=False, reduced=False):
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
        f_dir = self._get_folder_path(struct_scan, segm, auto_segm, reduced)
        
        # build file name and supply the file path based on current onedrive folder structure
        if struct_scan is not None:
            f_name = f"{f_name}_{struct_scan}"
            f_dir = os.path.join(f_dir, subj_file)
        
        if segm:
            f_name = f"{f_name}_segm"

        if auto_segm:
            f_name = f"{f_name}_automated_approx_segm"

        if reduced:
            f_name = f"{f_name}_cut"
            
        f_name = f"{f_name}.nii.gz"
        f_path = os.path.join(f_dir, f_name)
        
        return f_name, f_path
    
    def _get_folder_path(self, struct_scan=None, segm=True, auto_segm=False, reduced=False):
        if struct_scan is not None:
            return IMG_STRUCT_DIR
        
        if segm:
            return IMG_SEGM_DIR

        if auto_segm:
            return IMG_AUTO_SEGM_DIR

        if reduced:
            return IMG_AUTO_SEGM_REDUCED_DIR if auto_segm==True else IMG_REDUCED_DIR 