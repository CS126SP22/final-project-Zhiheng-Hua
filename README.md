# Image Classification App

## Overview
- This project is a machine learning project that builds a Cinder image classification app from scratch
- Algorithm: CNN (Convolutional Neural Network)

## Setup
- libraries:
  - [Cinder](https://libcinder.org/)  
  - [Eigen 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page)
  - [stb](https://github.com/nothings/stb/blob/master/stb_image.h)
- note: 
  - this project root directory should be placed in a subfoler of the cinder library
  - set the working directory of the app to the root directory of the repo to avoid file IO path issue

## How to use the App

### Main Page
<img src="demo-files\main-page.jpg" width="70%">

### Model Training and Saving
- Press `c` to configure params before dropping the dataset folder
  <br/>
  <img src="demo-files\config-page.jpg" width="70%">
- Drop an image dataset folder to start training. Dataset must follow structure specified in [util.h](include/util.h). It may take a while to train the data depending on dataset size. A training analysis graph will pop up on complete.
  <br/>
  <img src="demo-files\analysis-graph.jpg" width="70%">
- output `.cnn` model file will be saved to `\data` folder

### Uploading Model File 
- Model must have extension of `.cnn` with correct format
  <br/>
  <img src="demo-files\upload-model.jpg" width="70%">

### Make A Prediction
- A model must be present before making a prediction
  <br/>
  <img src="demo-files\warning.jpg" width="70%">
- Simply drop an image file into the app window 
  <br/>
  <img src="demo-files\prediction.jpg" width="70%">
  <img src="demo-files\prediction(2).jpg" width="70%">
  <img src="demo-files\prediction(3).jpg" width="70%">
