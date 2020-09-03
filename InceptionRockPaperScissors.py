import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf


import os

train_dir = r'C:\Datasets\rockpaperscissors\rps-train\train'
valid_dir = r'C:\Datasets\rockpaperscissors\rps-valid\validation'

train_dir_rock = os.path.join(train_dir, "rock")
train_dir_paper = os.path.join(train_dir, "paper")
train_dir_scissors = os.path.join(train_dir, "scissors")

train_rock_name = os.listdir(train_dir_rock)
train_paper_name = os.listdir(train_dir_paper)
train_scissors_name = os.listdir(train_dir_scissors)

local_weight_file = r'C:\PycharmProjects\ImageClassification\TransferLearningModel\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def visualize():

    rock_img = [ os.path.join(train_dir_rock, fname) for fname in train_rock_name[:4]]
    paper_img = [os.path.join(train_dir_paper, fname) for fname in train_paper_name[:4]]
    scissors_img = [os.path.join(train_dir_scissors, fname) for fname in train_scissors_name[:4]]

    nrow = 4
    ncol = 4

    ft = plt.gcf()
    ft.set_size_inches(50, 50)
    for i, img in enumerate(rock_img + paper_img + scissors_img):
        fig = plt.subplot(nrow*4, ncol*4, i+1)
        fig.axis('off')
        i = mpimg.imread(img)
        plt.imshow(i)
    plt.show()



def lossmetricgraph(history):

    T_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epoch = range(len(T_acc))

    plt.plot(epoch, T_acc, 'r', label='train acc')
    plt.plot(epoch, val_acc, 'b', label='val acc')
    plt.legend()
    plt.title('train vs val accuracy')

    plt.figure()

    T_acc = history.history['loss']
    val_acc = history.history['val_loss']

    plt.plot(epoch, T_acc, 'r', label='train loss')
    plt.plot(epoch, val_acc, 'b', label='val loss')
    plt.legend()
    plt.title('train vs val loss')

    plt.show()


def load_augument_data():

    train_gen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   zoom_range=0.2)

    train_generator = train_gen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    class_mode='categorical',
                                                    batch_size=128,
                                                    shuffle=True)

    valid_gen = ImageDataGenerator(rescale=1.0/255)

    valid_generator = valid_gen.flow_from_directory(valid_dir,
                                                    target_size=(150, 150),
                                                    class_mode='categorical',
                                                    batch_size=32)
    return train_generator, valid_generator


def loadInceptionModel():

    pretrainedmodel = InceptionV3(include_top=False,
                                  weights=None,
                                  input_shape=(150, 150, 3))

    pretrainedmodel.load_weights(local_weight_file)

    for layer in pretrainedmodel.layers:
        layer.trainable = False

    return pretrainedmodel

def append2model():

    model = loadInceptionModel()

    op_layer = model.get_layer('mixed7')
    op_layer_output = op_layer.output

    x = layers.Flatten()(op_layer_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(3, activation='softmax')(x)

    model_ap = Model(model.input, x)

    model_ap.compile(optimizer=RMSprop(lr=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['acc'])

    return model_ap



def makeCNNNetwork():
    l1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3))
    l2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    l3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
    l4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    l5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
    l6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    # Added more layers
    l5_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
    l6_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    # Added dropout layer to reduce overfitting
    l6_2_dp = tf.keras.layers.Dropout(0.5)

    l7 = tf.keras.layers.Flatten()
    l8 = tf.keras.layers.Dense(units=1024, activation='relu')
    l9 = tf.keras.layers.Dense(units=3, activation='softmax')

    model = tf.keras.models.Sequential([l1, l2, l3, l4, l5, l6, l5_1, l6_1, l6_2_dp, l7, l8, l9])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def prepareNtrain():

    model = makeCNNNetwork() #append2model()

    train, valid = load_augument_data()

    history = model.fit(train,
                        epochs=50,
                        steps_per_epoch=19,
                        verbose=1,
                        validation_data=valid,
                        validation_steps=11)

    model.save("CategoricalModel")

    lossmetricgraph(history)


#prepareNtrain()

from tensorflow.keras.preprocessing import image
def prediction():

    model = models.load_model("CategoricalModel")
    test_dir = r'C:\Datasets\rockpaperscissors\07_try-testing-the-classifier_rps-validation'
    test_name = os.listdir(test_dir)

    test_files = [ os.path.join(test_dir, x) for x in test_name]
    imag = np.random.randint(0, len(test_files))
    gta_image = test_files[imag]

    imgaa = image.load_img(gta_image,
                           target_size=(150, 150))

    img_array = image.img_to_array(imgaa)
    test_img = np.expand_dims(img_array, axis=0)
    op = model.predict(test_img)
    print(op)
    print(gta_image)

    bts = mpimg.imread(gta_image)
    plt.imshow(bts)
    plt.show()


prediction()

