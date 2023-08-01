import os
import numpy as np
from torch import stack, from_numpy
from torch.utils.data import Dataset
import nibabel as nib

# useful reference:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://blog.paperspace.com/dataloaders-abstractions-pytorch/


def create_ae_data_list(data_dir, outfile_str="data_list.txt"):
    """
    Given a target directory, generates an output file
    containing unique samples (patients) used in
    AutoencoderMRIDataset

    Inputs:
        data_dir - str path to data
        outfile_str - str name for output file (optional)

    Returns None
    """
    # obtain unique samples
    files = os.listdir(data_dir)
    files.sort()
    unique_files = []
    for curr_file in files:
        if curr_file.split('.')[-1] == 'gz':
            unique_files.append(curr_file.split('_')[0])
    unique_files = list(set(unique_files))
    unique_files.sort()
    print(f"Number of unique samples: {len(unique_files)}")

    # write out unique_files to disk
    with open(os.path.join(data_dir, outfile_str), 'w') as outfile:
        for item in unique_files:
            outfile.write("%s\n" % item)


def collate_fn(batch):
    """
    Provides support for custom object types during batch collation
    """
    return {
        "subj_no": [example["subj_no"] for example in batch],
        "vol": stack([example["vol"] for example in batch]),
        "affine": [example["affine"] for example in batch],
        "header": [example["header"] for example in batch],
    }


class AutoencoderMRIDataset(Dataset):
    """
    Class to create a custom on-demand data set for use
    in training an autoencoder on the 3D MRI volumes
    """
    def __init__(self, data_dir, data_list_fn, transforms=None):
        """
        Inputs:
            data_dir - string path to data
            data_list_fn - string file name listing data
        """
        self.data_dir = data_dir
        self.data_list_fn = data_list_fn
        self.transforms=transforms

        with open(data_list_fn, 'r') as file:
            self.n_data = sum(1 for _ in file)


    def __len__(self):
        return self.n_data


    def __getitem__(self, idx):
        if idx >= self.n_data:
            print(f"Warning: given index {idx} does not exist. Using first sample instead")

        # find a file corresponding to the current index
        with open(self.data_list_fn, 'r') as curr_file:
            for line_number, line_content in enumerate(curr_file):
                # load the first line as default
                if line_number == 0:
                    subj_no = line_content.split('.')[0].split('_')[0].strip()
                # set numbers according to index
                if line_number == idx:
                    subj_no = line_content.split('.')[0].split('_')[0].strip()

        # set data file names corresponding to 4 3D MRI modes
        data_file_0 = subj_no + '_11_T1' + '.nii.gz'
        data_file_1 = subj_no + '_11_T1GD' + '.nii.gz'
        data_file_2 = subj_no + '_11_T2' + '.nii.gz'
        data_file_3 = subj_no + '_11_FLAIR' + '.nii.gz'

        # load data file to image
        data_cur = nib.load(os.path.join(self.data_dir, data_file_0))
        # capture the affine transformation matrix
        affine = data_cur.affine
        # capture the image header
        header = data_cur.header
        # get the input dimensions
        n_h = data_cur.shape[0]
        n_w = data_cur.shape[1]
        n_d = data_cur.shape[2]
        # intialize an array of zeros with the same dimensions as the inpu
        # with an extra dimension to index the four input volumes
        vol = np.zeros((n_h, n_w, n_d, 4))

        # add data_cur to image
        vol[:, :, :, 0] = data_cur.get_fdata()

        # load remaining 3 MRI modes
        for ifile, file in enumerate(list((data_file_1, data_file_2, data_file_3))):
            vol[:, :, :, ifile + 1] = nib.load(os.path.join(self.data_dir, file)).get_fdata()


        # create a mapping of the sample
        sample = {
            "subj_no": subj_no,
            "vol": vol,
            "affine": affine,
            "header": header,
        }

        # apply any desired transformations
        if self.transforms:
            sample["vol"] = self.transforms(sample["vol"])

        return sample


class ToTensor:
    """
    converts numpy ndarrays to Tensors
    """
    def __call__(self, vol):

        # swap channels to be first axis
        # numpy array format HxWxZxC
        # PyTorch tensor format CxHxWxZ
        vol = vol.transpose((3, 0, 1, 2))
        return from_numpy(vol)
