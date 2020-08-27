# ResNet50-Smile-Classifier--CelebA-Dataset

"Synthetic Face ID project" requires training an alternative prediction model that will use the CelebA given
dataset annotation on the ResNet50 network to better determine features of synthesized images. We choose smile (expression) facial attributes for analysis. Note that we can always plug in more attributes easily as long as the attribute detector is available.

This model is trained to predict smile. Smile attribute is learnt as bi-classification problem with binary crossentropy loss, Adam optimizer (lr = 0.01) and Early Stopping as regularizer. As images produced by PGGAN and StyleGAN are with 1024×1024 resolution, we resize them to 224×224 before feeding them to this classification model.

The classifier will automatically divide the data into train validate and test sets. The classifier will use Keras's "Flow from Dataframe" method, as CelebA (training dataset) has both face images and annotation file.  

### Data-set
Some generative models create new synthetic faces using celeb faces. In order to train smile classifier, the CelebA data set will also be used for this study. CelebFaces Dataset (CelebA) adds a massive dataset featuring over 200 K images of celebrities, along with 40 attribute details. The images in this data set contain substantial variations in poses and scope. CelebA has a wide spectrum, vast numbers and high prices,
* 10,177 number of identities,
* 202,599 number of face images, and
* 5 landmark locations, 40 binary attributes annotations per image.


Github- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### Requirements 
* Python 3.7
* TensorFlow 2.2.0
* Keras 2.4.3
* Pillow
* MatPlotLib


### How To Run
- [x] Download CelebA dataset
     * Put Align&Cropped Images in ./celeba/*.jpg
     * Put Attributes Annotations in ./list_attr_celeba.txt
- [x] Download the ResNet50.py from this repository
- [x] In the console run:
     * python ResNet50.py
### References
[1] Andrew Ng. Deep learning course and specialization in convolutional neural network, 2018

[2] Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou. Deep Learning Face Attributes in the Wild. 2015
