# Cell-Nuclei-segmentation
This project was about image segmentation for cell nuclei through the kagle notebook

In this kernel I will be using U-Net model for segmentation of nuclei images.
U-Net model is commonly used in image segmentation problems. Links to get information about U-Net model - link1, link2
Aim of this kernel is to provide all the steps clearly which are required for segmenting the nuclei images.


1) We Need to Load all Libraries
2)Data Loading and data exploration
Let's understand the directory structure of the train dataset and how is data organised so that we can load the data from the required folder only.
So there are many subfolders in ../input/stage1_train/ folder which in turn contain ['masks', 'images'] subfolders.
The images and their corresponding masks have to be retrieved from ['masks', 'images'] subfolders.
Let's make a list of all subfolders in ../input/stage1_train/ so that we can retrieve images and corresponding images.


