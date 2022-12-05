# Final Project for CS 539 Machine learning: Pet finder application (https://sites.google.com/view/dkmax/home)
### Dennis Hofmann, Avantika Shrestha, Oluseun Olulana, Marcela Vasconcellos, Fatemeh Farajzadeh

Our goal is to develop a method that matches two images of the same pet. This would be extremely useful in a lost pet situation. If a person finds a pet, they can take a picture of them and upload it. Our method will convert that image into a latent vector that captures the pets core features. Our database is then queried for similar latent vectors. If a similar image is found, that image along with the contact information of the person that uploaded the image is returned. If a similar image is not found the image’s latent vector is stored in our database until the owner of the lost pet submits their image and the lost pet is reunited with its correct owner.

This project consists of 3 main parts. First, we prepare the dataset, then we train our model, and lastly, we test our method. We use this notebook file to prepare the dataset and test our method, but we use the remining python files in the directory to train our model. Due to the size of our dataset, training had to be conducted on a high-performance computing cluster.

1. <b>Preparing the data:</b>
For our dataset we are using the Dogs vs. Cats dataset from Kaggle (https://www.kaggle.com/c/dogs-vs-cats). This dataset consists of 25,000 images of both cats and dogs. To represent two different images of the same pet we augment the dataset by duplicating the dataset of images. Instead of 25,000 images we now have 50,000 where each pet has two images, one to represent the owner’s image and one to represent the image taken by the person that found the pet. Of course, we cannot assume that the pet owner and finder would take the exact same image, so we removed the background of all images, randomly added noise, randomly flipped images, and randomly rotated images.

2. <b>Model training:</b>
For our model we chose to leverage an autoencoder. Autoencoders have shown to work well at learning lower representations of images. Due to time and resource constraints our autoencoder is relatively simple with only a few convolutional layers. We also had to reduce the size of the training data to 10,000. To train our autoencoder we randomly selected 10,000 noisy and original images and fed them through the autoencoder with the goal of reducing the reconstruction error. Doing so allows the model to learn how to appropriately capture the import features of an image such that the image could be reconstructed. We trained the model for 200 epochs. The code for the autoencoder training can be found in the python files in the directory.

3. <b>Testing:</b>
To test our method, we first randomly select 2 images of the same pet from our testing dataset. We first upload the noisy image to represent the image taken by the person that found the pet, and then we upload the clean image to represent the image from the owner. We then observe if the two image’s latent vectors can be matched based on similarity. So far, we not observed any mistakes but as the database size continues to grow we may have to make our model more complex. 

There are two main steps needed for future work. First, we can further improve model training by tunning the current model or developing a new technique. Second, we can build an easy-to-use UI that makes our work more accessible to non-technical users. 

Training is set up to be done on GPU and ran on WPI's high-performance computing cluster (Turing). Notebook was ran on Google Colab and data was stored on Google drive. Ran on python 3.8 and please see requirments.txt for more information of packages.
