This document describes the steps necessary to install the 3D Slicer application to your computer, enable it to connect to Google Cloud Storage via the python interpreter, and execute the functions load_patient() and load_dataset().
<br> 
1) Download the installation file for 3D Slicer from https://download.slicer.org/ and install the application

2) Install the google-cloud-storage module to the 3D Slicer Python interpreter.  Because 3D Slicer uses its own embedded Python interpreter, the google-cloud-storage module must be installed to this interpreter via your system command prompt.

	WINDOWS:
	If you installed 3D Slicer to the default location, update the user name in the path and from the command prompt, run:  
		<code>"C:\Users\<your user name>\AppData\Local\NA-MIC\Slicer 5.2.2\bin\PythonSlicer.exe" -m pip install google-cloud-storage</code>	

	Otherwise, update the path according to where 3D Slicer is installed on your machine and run:  
		<code><path_to_3D_Slicer>\bin\PythonSlicer.exe -m pip install google-cloud-storage</code>

	MACOS/LINUX:  
    Update the path and from the command prompt, run:  
		<code><path_to_3D_Slicer>/Slicer --python-use-system-environment -m pip install google-cloud-storage</code>  
  
3) Run the 3D Slicer application.  Ensure the Python console is visible by selecting View-->Python Console

4) From the project GitHub repository, access "MRI_visualization.ipynb" 
	https://github.com/RiceD2KLab/BioCV_Su23/blob/95ef87618ccd4de32a502d03589431241e71ed0c/src/visualization/MRI_visualization.ipynb

5) Paste the functions load_patient() and load_dataset() into the Python console

6) Test the functions:

	a) Enter "load_patient()" in the Python console.  You should be prompted to enter an integer that corresponds to the patient whose data you would like to load.  For example, for patient "UPENN-GBM-00004_11", you should enter "4".  3D Slicer will load all imagining sequences and segmentations for the selected patient.

	b)  Enter "load_dataset()" in the Python console.  The list of datasets present within our Google Cloud Storage will appear. Input the integer corresponding to the dataset you would like to load. 3D slicer will load imaging sequences and segmentations for all patients included in that dataset. Note, currently this has only been tested on the "images_annot_reduced_norm" dataset. If other datasets have a different file structure, the function will need to be updated accordingly.
