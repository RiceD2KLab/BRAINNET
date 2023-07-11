# Source README

This readme explains our source code structure. The tree looks like:

```
src/
├── auth
│   ├── README.md
│   └── zinc-citron-387817-2cbfd8289ed2.json
├── models
│   └── autoencoder.py
├── notebooks
│   ├── 00_exploratory_data_analysis.ipynb
│   ├── 01_mri_data_prep_and_loading.ipynb
│   ├── 02_auto_segm_versus_segm.ipynb
│   ├── 03_prototype_conv_autoencoder.ipynb
│   ├── 04_cnn_cancer_classifier.ipynb
│   └── 05_maskformer_pilot_share_metrics.ipynb
├── README.md
├── utils
│   ├── data_handler.py
│   ├── get_python_requirements.ipynb
│   ├── google_storage.py
│   ├── make_autoencoder_dataset.py
│   ├── metrics.py
│   ├── mri_common.py
│   ├── mri_plotter.py
│   └── pyenv_import_all_reqs.ipynb
└── visualization
    ├── MRI_visualization.ipynb
    ├── README.md
    └── zinc-citron-387817-70b815932bac.json
```

## `src/auth`

This directory contains a secrets file `zinc-citron-387817-2cbfd8289ed2.json` used for accessing our Google storage as an alternative to using Microsoft OneDrive as it is persistent beyond the end of our time at Rice.

## `src/models`

This directory contains code for any custom models we implement. For classification and segmentation, we are currently fine-tuning  pre-trained models. For the data compression task of reducing the four structural MRI scans into three for input to these models, we have designed a 3D convolutional autoencoder whose class definition is contained within `autoencoder.py`.

## `src/notebooks`

These are our prototyping notebooks where we develop our ideas that will later be put into a standalone Python app. In specific:

### `00_exploratory_data_analysis.ipynb` 

This is our initial notebook where we explore the MRI data for the first time, experimenting with using Python packages `Nibabel` and `OneDriveDownloader`. This notebook further helps to inform some of our design choices in later notebooks.

### `01_mri_data_prep_and_loading.ipynb`

This notebook performs various data wrangling steps that prepares 2D slices from the 3D MRI volumes.

### `02_auto_segm_versus_segm.ipynb`

This notebook explores the relation between the expert reviewed annotation masks with those that were generated using the BrATS 2021 challenge-winning models as a performance baseline.

### `03_prototype_conv_autoencoder.ipynb`

This notebook tests training our 4-to-3 compressional 3D Convolutional Autoencoder and utilizes custom code from `utils/` and `models/`.

### `04_cnn_cancer_classifier.ipynb`

This notebook tests fine-tuning VGG19 for classification of 2D slices (cancer negative or cancer positive) and GradCAM for explainability.

### `05_maskformer_pilot_share_metrics.ipynb`

This notebook tests fine-tuning Meta's MaskFormer Vision Transformer model for segmentation, calculating metrics on the results.

## `src/utils`

This directory contains much of our custom Python (`.py`) modules that we have writen for re-usability across our different notebooks and eventually a standalone Python app.

### `data_handler.py`

This module ...

### `google_storage.py`

This module defines the class `GStorageClient` that is used to access our data hosted in a Google Storage bucket.

### `make_autoencoder_dataset.py`

This modules defines:

* function `create_ae_data_list` which generates a lit of unique patient sample IDs,
* class `AutoencoderMRIDataset` which inherits the `Dataset` class from `torch.utils.data`, implementing a custom `__getitem__` method for accessing data on-demand during batch training, and
* class `ToTensor` which converts the data stored as `numpy` array format (w, h, d, c) to tensor format (c, w, h, d) as expected by `torch`.

### `metrics.py`

This modules defines custom metrics used for evaluating our models' performance.

### `mri_common.py`

This module contains two dictionaries for mapping mask label integer values to string names and colors for visualization.

### `mri_plotter.py`

This module defines class `MRIPlotter` which ensures common parameters for all MRI plots.

Also note that there are two Jupyter notebooks within this directory:

* `pyenv_import_all_reqs.ipynb`
* `get_python_requirements.ipynb`

The first is a single-cell notebook that imports all packages we use. The second uses `pipreqsnb` to generate our [requirements.txt](../requirements.txt) file from the first notebook.

## `src/visualization`

This directory contains instructions and a notebook on using graphical tool 3D Slicer for visualizing MRI data. See the [Visualization README](visualization/README.md) for more details.



