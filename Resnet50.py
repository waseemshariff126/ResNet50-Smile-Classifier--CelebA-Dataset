# Common imports
import numpy as np
import os
import gc

import numpy as np
import pandas as pd


import tensorflow as tf
from keras.applications import resnet50

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.utils import layer_utils, to_categorical
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import scipy.misc
from matplotlib.pyplot import imshow

import warnings
warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0),kernel_regularizer = regularizers.l1_l2(l1= 0.01,l2 = 0.1))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X


def ResNet50(input_shape = (32, 32, 3), classes = 1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [256,256, 1024], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256,256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256,256, 1024], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)
    

    # output layer
    X = Flatten()(X)
    #X = Dense(1,activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    dense = Dense(1,activation='sigmoid')(X)
    dense = Dense(1,activation='softmax')(dense)
    
    # Create model
    model = Model(inputs = X_input, outputs = dense, name='ResNet50')

    return model

#-----------------------------------------------------------------------
#-------------------------CLASSIFICATION--------------------------------
#-----------------------------------------------------------------------
# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator
os.system('mkdir ./model/')
os.system('mkdir ./model/snapshot/')
os.system('mkdir ./model/output/')

csize = 224
batch_size = 64

import os
print(len(os.listdir("celeba")))

#=======================================================
# The following code describes how data is divided into training, validation and testing
#========================================================
df_path = "list_attr_celeba.csv"
df_split = pd.read_csv(df_path)

df_split = df_split.rename(columns={'image_id':'filename','Smiling':'class'})
df_split = df_split.replace(1,"smile")
df_split = df_split.replace(-1,"no_smile")
df1 = df_split[['filename','class']]
df2 = pd.read_csv('list_eval_partition.csv')
df2.columns = ['Filename', 'Partition']
df3 = df1.merge(df2, left_index=True, right_index=True)
df3.to_csv('celeba-Smiling-partitions.csv')
df4 = pd.read_csv('celeba-Smiling-partitions.csv', index_col=0)
del df4['Filename']
df4.loc[df4['Partition'] == 0].to_csv('celeba-Smiling-train.csv')
df4.loc[df4['Partition'] == 1].to_csv('celeba-Smiling-valid.csv')
df4.loc[df4['Partition'] == 2].to_csv('celeba-Smiling-test.csv')

df_train = pd.read_csv('celeba-Smiling-train.csv')
df_valid = pd.read_csv('celeba-Smiling-valid.csv')
df_test = pd.read_csv('celeba-Smiling-test.csv',skipinitialspace=True, skiprows=0, engine="python")

# Data preprpocessing
preprocess_input = resnet50.preprocess_input
train_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
    validation_split=0.25
)

# Data Generators 
train_generator = train_datagen.flow_from_dataframe(
    directory ='celeba/',
    dataframe=df_train,
    X_col = 'filename',
    Y_col = 'class',
    batch_size=64,
    target_size=(csize, csize),
    class_mode='binary',
    color_mode = 'rgb',
)
validation_generator = train_datagen.flow_from_dataframe(
    directory ='celeba/',
    dataframe=df_valid,
    X_col = 'filename',
    Y_col = 'class',
    batch_size=64,
    target_size=(csize, csize),
    class_mode='binary',
    color_mode = 'rgb',
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory='celeba/',
    x_col='filename',
    Y_col ='class',
    target_size=(csize, csize),
    batch_size=64,
    class_mode='binary',
    color_mode = 'rgb',
)

#=====================================================================================
# ResNet 50 model 
model = ResNet50(input_shape = (csize, csize, 3))
model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
#=======================================================================================

checkpointer = ModelCheckpoint(
		filepath='weights.h5',
		verbose=1, save_best_only=True	
	)

early_stopping = EarlyStopping(monitor='val_loss',patience=5)

# Model fit 
history = model.fit_generator(
    generator= train_generator,
    validation_data= validation_generator,
    epochs=5,
    callbacks=[checkpointer,early_stopping],
    steps_per_epoch=int(train_generator.n / train_generator.batch_size),
    validation_steps=int(validation_generator.n / validation_generator.batch_size)
)
# Save Model
tf.keras.models.save_model(model,'model/output/resnet_.h5')
# Evaluating Test Generator
test_generator.reset()
pred = model.evaluate_generator(test_generator, steps=int(test_generator.n / test_generator.batch_size),verbose=1)
print("Test Loss",pred[0])
print("Test Accuracy",pred[1])





