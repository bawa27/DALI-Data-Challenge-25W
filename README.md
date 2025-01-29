# DALI-Data-Challenge-25W
<p align="center">
<img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result5smallg.png" width="400" height="400"> <img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result7_big.png" width="400" height="400">
</p>

## Overview

This repository contains my submission for the 2025W DALI lab data science developer challenge. I was tasked with creating a program to detect barnacles in an image. My final model uses a transfer learning Mask R-CNN approach with Detectron2 trained 
on Microsoft's COCO dataset. Training images were largely generated through data augmentation via the albumentations package 
for Python. 


## Files and Dependencies
- **roi_extraction.ipynb** = Extracts the region of interest (ROI) from the original images and masks
- **data_augmentation.ipynb** = Creates 98 (can be increased or decreased) images of augmented data and their respective masks
- **coco_annotations.ipynb** = Generates json data containing the COCO required parameters for each image/mask
- **detectron2_maskrcnn.ipynb** = Trains, evaluates, and tests the model
- **output/** = Contains the final model and images outputs of some runs
- **plots/** = Contains plots of the training  loss
- **barnacle_dataset/** = Contains the training and testing images and masks - both augmented and original
- **initial_opencv_tests** = Contains first attempts at a solution with OpenCV algorithms such as contour detection, Hough circles, and watershed segmentation.

Requirements are given in the requirements.txt file, but the main dependencies are:
- OpenCV
- PyTorch (TorchVision and Torch)
- Detectron2
- Albumentations

## The Design Process
Before writing any code, I decided to try and break down how the problem could be solved. I started with pen and paper, jotting down any ideas of a hypothetical system that these scientists could use to detect barnacles. 
I believe that models are best utilized alongside humans, instead of totally replacing them, so I decided that a tool that could detect barnacles and then allow the user to edit the results would be the best solution. 
Perhaps the marking of false positives or negatives could be used to improve the model. Of course, the subtask I would want for this project would be to create a model that could detect barnacles with a high degree of accuracy.
Some other required subtasks could be image analysis, a features team, and a user interface team. Image analysis would be particularly important for generating data that can be used to determine configuration settings for training and testing.

I knew the primary issue was going to be the limited dataset, so my first idea was to use a purely traditional approach. 
I haven't done too much with OpenCV so I also decided to take it as a learning experience. My first contour detection test made me realize that one major issue with the barnacles was the pattern on their shells. 
These extra patterns meant extra edges and oversegmentation. Additionally, the density of the barnacles led to many patches being counted as only one barnacle. 
The best solution I found to this was to use a watershed segmentation algorithm with more Gaussian Blurring, which essentially makes a topographic map of a region and flood fills it 
to find overlapping shapes. The blurring would help with the textures. I did get somewhere with this, to the point where I could likely tweak the parameters until I had a relatively acceptable loss, but I wasn't satisfied. 
I was missing entire areas of the ROI, and I felt as if the watershed algorithm was the upper limit of what could be accomplished with traditional image processing.

I then decided to move on to a more modern approach. I had heard of Mask R-CNN before, and I knew it was a powerful tool for instance segmentation. 
I decided to use Detectron2, a Facebook AI Research library that is built on PyTorch. 

## The Final Model
Detectron2 has 5 essential steps to train a model: feature extraction, region proposal network, ROI align, prediction head, and post-processing. Feature maps are extracted by a backbone CNN, in this case ResNet50, 
and the Feature Pyramid Network (FPN) enhances these maps at different resolutions for better small-object detection. The RPN proposes regions of interest, which are then aligned to a fixed size and shape by ROI align. 
Only the top N proposals are passed to ROI align. The prediction head then predicts the class, bounding box, and mask head of the proposal. The mask head is generated with 
a Fully Convolutional Network (FCN). Finally, post-processing is done to remove overlapping boxes and non-maximum suppression is applied.

I chose to use Detectron2 over more modern models like YOLO because of its performance with instance segmentation. 
Modern YOLO versions have added support for segmentation, but they don't quite have the same support for an algorithm like Mask R-CNN. I figured precision was more important than speed in this case,
especially due to how many small, overlapping barnacles there were.

Because we were only provided with 2 images, I used data augmentation heavily to increase the size of the dataset. I only increased to 100 images, as I was running out of time and resources, 
but more images can be made as well. Transfer learning was also used to speed up the training process, as training a model from scratch would have taken too long and could result in significant overfitting
due to the presence of essentially only augmented training images. The model was trained for 2000 iterations with a learning rate of 0.00025 on the model seen now.
Final total losses were as low as 1.18, or <0.15 object loss. Approximate barnacle counts were ~130 for the smaller test image and ~750 for the large one. Some test images are provided below, and some more are at the top of this README.

<p align="center">
<img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result5smallg.png" width="400" height="400">  <img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result7_big.png" width="400" height="400">
</p>


## Challenges and Potential Improvements 

Despite its performance, the model is not perfect. Due to limited training data and lower parameter values to conserve resource, the model is not as good as it could be.
Working around limited computational resources was a significant challenge, as training the model took a long time and required a lot of memory. I came up with a few inventive solutions to this problem,
but they didn't quite pan out (these are described later). To deal with running locally on my laptop, I lowered things like batch size and iterations, which could have a marked impact on the model's performance. 
I also reduced the size of the training set due to memory and time constraints. 

Besides compute, I also encountered trouble with overfitting. During my final runs, I found that variance in the loss curves from overfitting was growing no matter how low the learning rate was, which led me to believe that
the model was compounding on top of previous versions rather than learning with new configurations every run. Below are the loss curves of the first, fourth, and seventh runs respectively. The learning rate for each was 0.00025,
0.00015, and 0.00005, yet note the increase in varianceTo combat this, I reverted the repo back to the model with the best performance. Although there is still overfitting, the model
works quite well on the test images. To combat compounding data, I would create a few lines of code to delete the previous model and data before training a new one.

<p align="center">
<img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/plots/loss_curve1.png" width="300" height="300"> <img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/plots/loss_curve4.png" width="300" height="300"> <img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/plots/loss_curve7.png" width="300" height="300">
</p>

Even with the high performance of some configurations, I found that the model has trouble with scaling as well. The two test images show barnacles at very different scales - one zoomed in and one zoomed out. The model can perform well on both scales, but editing the configuration is required.
Specifically, INPUT.MIN_SIZE_TEST must be changed to 0 for zoomed out images and 250 for zoomed in ones for the best performance. Below I show the difference between a value of 0 (right) and 250(middle) for the zoomed in image. The zoomed out image is shown on the right.
This is more easily changed with my test set specifically, as I only have two images, but would present a problem on a larger pipeline. An idea I had to combat this was to create a function that would automatically detect the scale of the image and adjust the configuration accordingly.

<p align="center">
<img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result5_smallb.png" width="300" height="300"> <img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result5smallg.png" width="300" height="300"> <img src="https://github.com/bawa27/DALI-Data-Challenge-25W/blob/main/output/images/result5_large.png" width="300" height="300">
</p>

One of the greatest missteps of this project was not creating a separate validation dataset. With this, overfitting could be detected more easily and model metrics could be more precise.
For this quick proof of concept it isn't entirely necessary, but for a more robust model it would be essential. It would also be helpful to monitor the model's performance on the train and validation images over time to see if the model is improving or not.

Finally, for a larger-scale project I would edit the code to be more modular. This was a sort of "quick and dirty" solution to get to a final model, but a more sophisticated pipeline would be necessary to handle more data. 

## Conclusions

I believe the model I have created is a good proof of concept for a more robust model. It performs well on the test images and has a relatively 
low loss for the resources it was given. Unfortunately the compounding overfitting is an extreme problem for this model, 
and I believe the model could have been even better had the final model been reset before every run. However, with a more robust pipeline, 
this model could be improved upon and used in a real-world setting, and I think it would be worth doing so.

## Learning Outcomes

Personally, I never want to look at a barnacle again ... 

But really, I had an amazing time with this project, which is why I didn't just stop on my first model. Im glad I decided to use a SOTA model,
since I have never done machine learning quite like this before. I'm more used to standard linear regression or SVM models, so this was a fun challenge. 
By far my favorite thing that I learned about was Dartmouth HPC. I initially tried to remotely access my PC at home, but Detectron2 requires an
NVIDIA GPU, which I don't have. One of my friends joked that I should ssh into an HPC cluster, and I was surprised to find out that Dartmouth had a few. I was even more 
surprised to find out that they are free to access by anyone with a Dartmouth NetID. I got as far as loading my code in a jupyter notebook on the cluster and connecting
to it with my lapop, but I realized that only the best cluster, Discovery, has a GPU. Discovery also, unfortunately, requires jobs to be scheduled and in a specific format.
I decided against using just the CPU on the cluster that I was on and instead just continued running the model locally.
It was definitely overkill to attempt to use an HPC cluster for this challenge, but I thought it would be a fun twist. I will definitely be using it in the future for more computationally intensive projects.
