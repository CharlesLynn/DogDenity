# ![graph1](doc/logo.png)

## Overview
DogDentity is a two-week Data Science project, by Charles Lynn, done during the Galvanize immersive program. The goal of this project was to explore the proficiency of convolutional neural networks at identifying variations within canine breeds by images. DogDentity includes 36 unique breeds* and has validation accuracy of 47.8%; random guessing is 2.7%. Over 31,000 images scraped from image-net.org were used in training the neural network. Techniques such as image augmentation, further prevented overfitting and increase validation accuracy.

## Convlutional Neural Network
![CNN](doc/CNN2.png)
Looking at pixels individually has little to no valuable information for person or a neural network. Pixels must be looked at togeather in groupings to have distgunible features or values. This is why I used a Convutional Neural Network, the convlutional layers process portions of the input image, called receptive fields. From these fields the neural network can learn features that generalize to images outsite the training set. 

## Results
![CNN](doc/graph1.png)
- 36 Breeds (most popular)
- Validation accuracy of 47.8%
- Random guessing is 2.7%
- Image Agumentaions: Rotation, Zooming, Horizontal flipping

Image Agumentaion drastically reduced overfitting. Dropout was not necessary.

## Future Work
Clean data is important, many of the collected images have extraneous background information that creates noise when training the model. By cropping the dataset using a trained OpenCV classifier, the model's accuracy could be significantly increased.
<p align="center">
  <img align="middle" src="doc/opencv-python.jpg" alt="opencv-python">
  <img align="middle" src="doc/crop.png" alt="cv2crop">
</p>


## Technologies Used
- [python](https://www.python.org/)
- [Keras](http://keras.io/)
- [Theano](http://deeplearning.net/software/theano/)
- [OpenCV](http://opencv.org/)
- [AWS EC2](https://aws.amazon.com/)


## Website Demo 
<p align="center">
  <img align="middle" src="doc/website.png" alt="website1" height="300" width="400">
  <img align="middle" src="doc/website2.png" alt="website2" height="300" width="400>
</p>

- API by Charles Lynn, Web Design by Chris Castro.
- [DogDenity API GitHub](https://github.com/CharlesLynn/DogDenity_API)
- [DogDenity Website (Under Construction!)](http://54.205.134.57:5000/static/dogdentity/public/index.html)

## *Included breeds:
Basset Hound, Husky, Beagle, King Charles Spaniel, Bernese Mountain Dog, Labrador, Border Collies, Mastiff, Boston Terrier, Minature Schnauzer, Boxer, Newfoundlands, Brittany Spaniel, Pointer Shorthaired, Chihuahua, Pomeranian, Cocker Spaniel, Poodle, Corgi, Pug, Dachshund, Rhodesian Ridgeback, Doberman, Rottweiler, English Bulldog, Shetland Sheepdog, French Bulldog, Shih Tzu, German Shepherd, Vizsla, Giant Schnauzer, Weimaraner, Golden Retrievers, West Highland White Terrier, Great Dane, Yorkshire Terrier.
