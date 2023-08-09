import os

from google.cloud import storage as gcs

# Google drive authentication
STORAGE_BUCKET_NAME = "rice_d2k_biocv"
STORAGE_AUTH_FILE = os.path.join("auth", "zinc-citron-387817-2cbfd8289ed2.json")

class GStorageClient:
    """
    Creates an GStorageClient object for accessing remote google storage bucket

    Requires an authentication file saved as .json format at src/auth
    """
    def __init__(self):
        # authenticate storage client if specified. if not, runtime will automatically be used
        self.storage_client = gcs.Client.from_service_account_json(STORAGE_AUTH_FILE)

    def file_exists(self, source_path):
        """
        Checks whether a file exists

        Args:
            source_path (str): path to file on remote storage

        Returns:
            bool
        """
        blob = self._get_blob_client(source_path)
        return blob.exists()

    def download_blob_as_file(self, source_path, destination_path):
        """
        Downloads an entity from remote storage as regular file type

        Args:
            source_path (str): path to file on remote storage
            destination_path (str): path to local storage

        Returns:
            None
        """
        # save the blob into the temp file path
        blob = self._get_blob_client(source_path)
        blob.download_to_filename(destination_path)

    def download_blob_as_bytes(self, source_path):
        """
        Downloads an entity from remote storage as byte type
        Does not save locally, but assigns to a variable in runtime

        Args:
            source_path (str): path to file on remote storage

        Returns:
            byte stream of file
        """
        blob = self._get_blob_client(source_path)
        return blob.download_as_bytes()

    def save_from_source_path(self, source_path, destination_path):
        """
        Uploads a file from local to remote storage

        Args:
            source_path (str): path to file on local storage
            destination_path (str): path to file on remote storage

        Returns:
            None
        """
        # Upload temp_file_path to blob path
        blob = self._get_blob_client(destination_path)
        blob.upload_from_filename(source_path)

    def save_text(self, destination_path, text):
        """
        Uploads text-like data to remote storage

        Args:
            destination_path (str): path to file on remote storage
            text (str): text-like data to be saved

        Returns:
            None.
        """
        blob = self._get_blob_client(destination_path)
        blob.upload_from_string(text)

    def list_blob_in_dir(self, prefix):
        """
        Get a listing of entities in remote storage directory

        Args:
            prefix (str): directory on remote storage to obtain file listing

        Returns:
            list of strings
        """
        bucket = gcs.Bucket(self.storage_client, STORAGE_BUCKET_NAME)
        if not prefix.endswith("/"):
            prefix += "/"

        # list all files in the directory
        blob_paths = self.storage_client.list_blobs(bucket, prefix=prefix)
        blobs = [os.path.basename(blob.name) for blob in blob_paths]

        # probably a mac issue. file is created when moving files to google cloud.
        # just delete it from the list of files
        if ".DS_Store" in blobs:
            blobs.remove(".DS_Store")

        return blobs

    def _get_blob_client(self, blob_path):
        """
        Helper function to get entity path on remote client

        Args:
            blob_path (str): path to entity

        Returns:
            string
        """
        bucket = self.storage_client.get_bucket(STORAGE_BUCKET_NAME)
        blob = bucket.blob(blob_path)
        return blob
