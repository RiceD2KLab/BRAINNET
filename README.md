# Rice D2K DSCI 535 Capstone Summer 2023, TeamCV Biomedical Imaging

![Project Splash Image](/img/general/project_splash_img.png)

## Team Members
Huafeng Liu, Ben Dowdell, Todd Engelder, Nicolas Oso, Keith Pulmano, Zida Wang


## Description
Our team's capstone project focuses on the application of current state-of-the-art (SOTA) computer vision models to 3D MRI volumes of patients diagnosed with Glioblastoma cancer. Glioblastoma (GBM) is a very aggressive form of cancer, and "survival rates and mortality statistics for GMB **have been virtually unchanged** for decades." [Data source: BrainTumor.org](https://braintumor.org/events/glioblastoma-awareness-day/about-glioblastoma/)

[![Glioblastoma Stats](img/general/gbm_stats.png)](https://braintumor.org/events/glioblastoma-awareness-day/about-glioblastoma/)

[![5-year survival rates](img/general/cancer_5_year_survival_rates.png)](https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2023/2023-cancer-facts-and-figures.pdf)

Our motivations include:

1. Advance diagnostic, treatment, and medical research
2. Test novel data science tools on 2D and 3D image data
3. Support exploration applications in the energy industry (image data idea transfer, such as 3D seismic data)

 Our objectives are to develop machine learning tools focused on:

1. Automated Brain Tumor Segmentation
2. Biomaker prediction (classification)
3. Brain Tumor detection (classification / anomaly detection)

## Data
We are using a data set from the Cancer Imaging Archive for segmentation and classification of brain MRI scans for patients diagnosed to have Glioblastoma-type cancer. The data is hosted [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642) and is collected and made available by the University of Pennsylvania Health System.

**Data citation**:

Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., … Davatzikos, C. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)

**Publication citation**:

Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., Lustig, R., Desai, A. S., Bagley, S. J., Mourelatos, Z., Morrissette, J., Watt, C. D., Brem, S., Wolf, R. L., Melhem, E. R., Nasrallah, M. P., Mohan, S., O’Rourke, D. M., Davatzikos, C. (2022). The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics. In Scientific Data (Vol. 9, Issue 1). [https://doi.org/10.1038/s41597-022-01560-7](https://doi.org/10.1038/s41597-022-01560-7)

**TCIA citation**:

Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)

The data set is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

This data set includes:

1. 3D mpMRI Images
    * T1, T2, T1-GD, and T2-Flair attribute volumes (~630 patients of 1-5+ GB each, 800,000+ images)
    * DICOM and NIFTI format
    * Cancerous and non-cancerous instances
    * Additional dataset of MRI images from normal, healthy patients (600 images with T1, T2, and PD-weighted attributes)
2. Brain Tumor annotations
    * de novo Glioblastoma tumors
    * 3D volumes for multiple patients
3. Histopathology slides
    * RGB images
4. Clinical data
    * survival dates after surgery
    * MGMT (Unmethylated, Indeterminate, Methylated)
    * IDH1
    * Demmographics

## Data Science Pipeline

1. 3D Images
2. Wrangled Images
3. ML model
4. Label Predictions
5. Performance Evaluation