import os
import tempfile

from google.cloud import storage as gcs

# Google drive authentication
STORAGE_BUCKET_NAME = "rice_d2k_biocv"
STORAGE_AUTH_FILE = os.path.join("auth", "zinc-citron-387817-2cbfd8289ed2.json")

class GStorageClient:
    def __init__(self):
        # authenticate storage client if specified. if not, runtime will automatically be used
        self.storage_client = gcs.Client.from_service_account_json(STORAGE_AUTH_FILE)
        
        
    def download_blob_as_file(self, blob_path):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            temp_file_path = temp_file.name

        # get bucket instance
        bucket = self.storage_client.get_bucket(STORAGE_BUCKET_NAME)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(temp_file_path)
        return temp_file_path
    
    def save_to_blob(self, blob_path, temp_file_path):
        bucket = self.storage_client.get_bucket(STORAGE_BUCKET_NAME)

        # Create a Blob object in the bucket
        blob = bucket.blob(blob_path)
        # Path of the local file
        blob.upload_from_filename(temp_file_path)
        
    def list_blob_in_dir(self, prefix):
        bucket = gcs.Bucket(self.storage_client, STORAGE_BUCKET_NAME)
        
        # list all files in the directory
        blob_paths = self.storage_client.list_blobs(bucket, prefix=prefix)
        blobs = [os.path.basename(blob.name) for blob in blob_paths]
        
        # probably a mac issue when moving files to google cloud. just delete it from the list of files
        if ".DS_Store" in blobs:
            blobs.remove(".DS_Store")
                
        return blobs