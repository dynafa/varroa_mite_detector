# varroa_mite_detector
# ITML7.110 - Varroa destructor object detection application
![Tensorflow](../images/Tensorflow.jpg)

![Varroa](../images/Adult-bee-with-varroa-mites-2.png)

##Introduction

The parasitic mite Varroa destructor is the greatest single driver of the global honey bee health decline. Better understanding of the association of this parasite and its host is critical to developing sustainable management practices.

### Varroa mite locations

The following infographic illustrates an interesting 

![Varroa Locations](../images/Preferred_feeding_location_of_Varroa_destructor_mites_on_adult_host_bees.jpg)

###  considerations


[Source](https://www.pnas.org/content/116/5/1792)

##Project objectives and development processes:
**Project objectives**
- Identify and localize 4 classes of objects in images and video
    - Queen bee
    - Worker bee
    - Drone bee
    - Varroa mite
- Use the data collected from use of the AI application to quantify the infection of a bee colony with varroa mite

**Rough plan for going about creating the application:**

1. Collect data required for training using GoogleImageDownload
2. Label the different data types
3. Create a function to store the data into HDF5 format using hpy5
4. Create a function to load that data from HDF5 format into the program and split it into training and test data sets
5. Create a sequential neural network for the data to pass through.
6. Initially create a simple DNN for image classification with emphasis on the data storage format
7. Bring in CNN concepts to achieve object detection.

**Project dependancies**
- tensorflow (2.0.0)
- hpy5
- matplotlib




**File directory structure**
```markdown
.
├── images
│   └── dronebees
│       ├──1.jpg
│       ├──2.jpg
│   ├── workerbees
│   └── queenbees
├── annotations
│   ├── 
│   ├── 
│   ├── 
│   └── 
├── models
├── hdf5_datasets
├── training_1
│   ├── 
│   ├── 
│   ├── 
│   └── 
├── documentation
├── training.py
└── eval.py

```

**Resources:**

[Quick start for using HDF5 with Python](https://h5py.readthedocs.io/en/stable/quick.html)

[Pretrained model with transfer learning tutorial](https://towardsdatascience.com/custom-object-detection-using-tensorflow-from-scratch-e61da2e10087)

[Google image download automation tool](https://github.com/hardikvasa/google-images-download)

[Image annotation tool](https://github.com/tzutalin/labelImg)

[SSD Object detection with TF Tutorial](https://lambdalabs.com/blog/how-to-implement-ssd-object-detection-in-tensorflow/)

[OpenCV Basic Object Detection Tutorial](https://www.pyimagesearch.com/2019/05/13/object-detection-and-image-classification-with-google-coral-usb-accelerator)

[Yolo Object Detection Tutorial](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv)

[Research paper on object recognition](http://www.joig.org/uploadfile/2018/0717/20180717055805469.pdf)

[Industry information on conventional varroa mite detection and control methods](https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/agriculture-and-seafood/animal-and-crops/animal-production/bee-assets/api_fs222.pdf)
