# Rock-Paper-and-Scissors

# Objective

- This project belongs to [kaggle's competitions](https://www.kaggle.com/c/rock-paper-scissors/overview) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the second course in this specialization. 

- Rock, Paper and Scissors is a dataset containing 2,892 images of diverse hands in Rock/Paper/Scissors poses.
-
- The objective of this study is to correctly identify if the image.

# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, seaborn, matplotlib, tensorflow, keras, os. 

# Data description  

- Rock Paper Scissors contains images from a variety of different hands, from different races, ages and genders, posed into Rock, Paper or Scissors and labelled as such. These images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images. Each image is 300×300 pixels in 24-bit color.  

- The images looks like:
  <p align="center">
   <img src="https://github.com/lilosa88/Rock-Paper-and-Scissors/blob/main/Images/Screenshot%20from%202021-05-22%2008-41-11.png" width="300" height="480">
  </p> 
  
# Feature engineering

- We define each directory using os library.

- Use of data generators. It read the pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We have one generator for the training images and one for the validation images. The two generators yield batches of images of size 150x150 and their labels (binary). 

- Data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. In our case, we will preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

- Imagen Augmentation (only in the second and third model) which is a very simple, but powerful tool to help to avoid overfitting. To put it simply, if you are training a model to spot cats, and your model has never seen what a cat looks like when lying down, it might not recognize that in future. Augmentation simply amends your images on-the-fly while training using transforms like rotation, among others.

# Neural Network model

- This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:
  - Four Convolution layers with a corresponding MaxPooling layer each one which is then designed to compress the image, while maintaining the content of the           features that were highlighted by the convlution.
  - One flatten layer: It turns the images into a 1 dimensional set.
  - One Dropout layer: with 50% of the neurons that will be removed.
  - Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer           consisted in 512 neurons with relu as an activation function. The second, have 3 neurons and softmax as activation function. 

- We built this model using RMSprop optimizer and categorical_crossentropy as loss function.

- The number of epochs=25

- We obtained Accuracy 0.9806 for the train data and Accuracy 0.9624 for the validation data. 

 <p align="center">
   <img src="https://github.com/lilosa88/Rock-Paper-and-Scissors/blob/main/Images/Screenshot%20from%202021-05-22%2008-41-55.png" width="360" height="300">
  </p> 
  
