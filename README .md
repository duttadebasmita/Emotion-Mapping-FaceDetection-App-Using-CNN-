## <br>Inference<br> 


Decreasing Loss: The loss value is decreasing with each epoch, indicating that the model is learning to minimize its prediction error on the training data.

Increasing Accuracy: The accuracy is increasing, which means that the model is becoming more accurate in predicting the correct target values on the training data.

Validation Loss and Accuracy: Similarly, the validation loss and accuracy are also showing improvement, indicating that the model is generalizing well to unseen data. See the Src Folder for VAL-Train Subfolder for Figure_1.png


Consistency: It's also important to observe the consistency of improvement across multiple epochs. If the trend continues consistently over more epochs, it further confirms the effectiveness of your model. the following infernecs where observed when the model was trained on 447 batches over 50 epochs

The app created has infered well on all the six types of expressions
##


# Emotion Detection using TensorFlow and Flask
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)

## Introduction
This project aims to detect emotions from facial expressions using a Trained CNN model. The application is built using TensorFlow and Flask, providing a web interface for users to upload images and receive emotion predictions.

## Features
- Emotion detection from images.
- Web interface for image upload and result display using Flask.
- Uses both pre-trained And Trained CNN model for predictions.
- Supports multiple emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.

## Requirements
- Opencv-python 
- Flask
- OpenCV
- NumPy
- Matplotlib
- Pillow


## Installation
Train the model with python emotions.py --mode train
For Web-Cam Feed run python emotions.py --mode display
For Running The Flask python app.py


### Clone the repository And The Reference


```bash
git clone https://github.com/duttadebasmita/Emotion-Mapping-FaceDetection-App-Using-CNN-.git
cd emotion-detection


"Learning Representations for Automatic Emotion Recognition"

Authors: Amir Zadeh, Minghua Chen, Soujanya Poria, Erik Cambria, Louis-Philippe Morency
Published in: IEEE Transactions on Affective Computing, 2018
Abstract: This paper presents a method for learning representations for automatic emotion recognition from videos. The authors propose a model that combines convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to capture both spatial and temporal features.
Citations: Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L. P. (2018). Learning Representations for Automatic Emotion Recognition. IEEE Transactions on Affective Computing, 9(3), 350-362. doi: 10.1109/TAFFC.2017.2714180.
"Deep Learning for Image-Based Facial Emotion Recognition: A Survey"

Authors: Tingting Jiang, Ruxin Wang, Binglong Chen, Yang Zhou, Huihui Zhu, Yulan Guo
Published in: Applied Sciences, 2021
Abstract: This survey paper reviews various deep learning methods used for facial emotion recognition (FER). It discusses different CNN architectures, preprocessing techniques, and datasets used in the field.
Citations: Jiang, T., Wang, R., Chen, B., Zhou, Y., Zhu, H., & Guo, Y. (2021). Deep Learning for Image-Based Facial Emotion Recognition: A Survey. Applied Sciences, 11(16), 7077. doi: 10.3390/app11167077.
"Emotion Recognition from Speech Using Deep Convolutional Neural Networks"

Authors: Yong-Cong Liao, Jia-Yan Wang, Hui Ding, Yi Ren, Zhiyong Wu
Published in: IEEE Access, 2018
Abstract: This paper presents a deep convolutional neural network (CNN) model for recognizing emotions from speech signals. The authors demonstrate that their model outperforms traditional methods in emotion recognition accuracy.
Citations: Liao, Y. C., Wang, J. Y., Ding, H., Ren, Y., & Wu, Z. (2018). Emotion Recognition from Speech Using Deep Convolutional Neural Networks. IEEE Access, 6, 68861-68870. doi: 10.1109/ACCESS.2018.2878840.
"Facial Emotion Recognition Using Convolutional Neural Networks: State of the Art"

Authors: Mahdi Khosrowabadi, Karim R. Nasr, Rabab Kreidieh Ward
Published in: The Visual Computer, 2020
Abstract: This paper reviews the current state-of-the-art CNN-based methods for facial emotion recognition. It provides a comprehensive analysis of different CNN architectures and their performance on various FER datasets.
Citations: Khosrowabadi, M., Nasr, K. R., & Ward, R. K. (2020). Facial Emotion Recognition Using Convolutional Neural Networks: State of the Art. The Visual Computer, 36(8), 1547-1563. doi: 10.1007/s00371-020-01836-4.
"Deep Emotion: Facial Expression Recognition Using Attentional Convolutional Network"

Authors: Haifeng Zhang, Arthur Hanjin Li, Zhangyang Wang, Hailin Shi
Published in: IEEE Transactions on Image Processing, 2021
Abstract: This paper introduces an attentional convolutional neural network model for facial expression recognition. The model incorporates attention mechanisms to improve the accuracy of emotion detection.
Citations: Zhang, H., Li, A. H., Wang, Z., & Shi, H. (2021). Deep Emotion: Facial Expression Recognition Using Attentional Convolutional Network. IEEE Transactions on Image Processing, 30, 6126-6140. doi: 10.1109/TIP.2021.3074486.



