import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np

train_dir = r"C:\Datasets\dogs-vs-cats\train"
train_cats_images_dir = os.path.join(train_dir, "cats")
train_dogs_images_dir = os.path.join(train_dir, "dogs")
train_cats_names = os.listdir(train_cats_images_dir)
train_dogs_names = os.listdir(train_dogs_images_dir)

validation_dir = r"C:\Datasets\dogs-vs-cats\validation"
validation_cats_images_dir = os.path.join(validation_dir, "cats")
validation_dogs_images_dir = os.path.join(validation_dir, "dogs")


def watchPics():
    nrows = 4
    ncols = 4
    index = 8

    fig = plt.gcf()
    fig.set_size_inches(nrows * 3, ncols * 3)

    cats_pics = [os.path.join(train_cats_images_dir, fname) for fname in train_cats_names[:index]]
    dogs_pics = [os.path.join(train_dogs_images_dir, fname) for fname in train_dogs_names[:index]]

    print(cats_pics)
    print(dogs_pics)

    for i, image in enumerate(cats_pics + dogs_pics):
        sg = plt.subplot(nrows, ncols, i + 1)
        img = mpimg.imread(image)
        plt.grid(False)
        sg.axis('off')
        plt.imshow(img)
    plt.show()


# watchPics()


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
    l8 = tf.keras.layers.Dense(units=512, activation='relu')
    l9 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    model = tf.keras.models.Sequential([l1, l2, l3, l4, l5, l6, l5_1, l6_1, l6_2_dp, l7, l8, l9])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def lossMetricsGraph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epoch = range(len(acc))

    plt.plot(epoch, acc, 'bo', label='training accuracy')
    plt.plot(epoch, val_acc, 'b', label='validation accuracy')
    plt.title('accuracy vs val_accuracy')
    plt.legend()

    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(epoch, loss, 'bo', label='training loss')
    plt.plot(epoch, val_loss, 'b', label='validation loss')
    plt.title('loss vs val_loss')
    plt.legend()
    plt.show()


def image_augmentation():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    return train_datagen


def prepareAndTrain():
    train_datagen = image_augmentation()  # Data augmentation params are added to reduce over fitting
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(150, 150),
                                                        batch_size=128,
                                                        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                  target_size=(150, 150),
                                                                  batch_size=50,
                                                                  class_mode='binary')



    # training
    model = makeCNNNetwork()
    history = model.fit(train_generator,
                        steps_per_epoch=16,  # calculated by images = batch * step
                        epochs=5,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=10)

    model.save('CatsAndDogsBinaryClassifierModelGPU')

    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='upper right')
    # plt.show()

    lossMetricsGraph(history)


#prepareAndTrain()


def predict():
    model = tf.keras.models.load_model(r"C:\Users\rsaroj\PycharmProjects\NLPTensorflow\textgen-model2") # loading Inception Model trained with Transfer learning
    # test_dir = r"C:\Datasets\dogs-vs-cats\test1\test1"
    # testName = os.listdir(test_dir)
    #
    # test_image_path = os.path.join(test_dir, testName[random.randint(0, len(testName) - 1)])
    # print(test_image_path)
    #
    #
    # img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(150, 150))
    # img_arr = tf.keras.preprocessing.image.img_to_array(img)
    # image_test = np.expand_dims(img_arr, axis=0)

    # pred_op = model.predict(image_test)
    #
    # if pred_op[0] > 0.5:
    #     print("cat: " + str(pred_op[0]))
    # else:
    #     print("dog: " + str(pred_op[0]))
    #
    # img2 = mpimg.imread(test_image_path)
    # plt.imshow(img2)
    # plt.show()


predict()


def checkLayersWeLisualize():
    model = tf.keras.models.load_model("CatsAndDogsBinaryClassifierModel")
    op_layers = [layer.output for layer in model.layers]
    op_layers_names = [layer.name for layer in model.layers]

    # print(len(op_layers))
    # print(len(op_layers_names))
    # print(op_layers[0].shape)

    wesualize_model = tf.keras.models.Model(inputs=model.input, outputs=op_layers)

    # Get Image Randomly
    test_dir = r"E:\Study ML DataSets\dogs-vs-cats\test1\test1"
    testName = os.listdir(test_dir)
    test_image_path = os.path.join(test_dir, testName[random.randint(0, len(testName) - 1)])
    print(test_image_path)

    img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(300, 300))
    image_test = tf.keras.preprocessing.image.img_to_array(img)

    img_arr = np.expand_dims(image_test, axis=0)

    op = wesualize_model.predict(img_arr)
    print(op[0].shape)

    for layer, name in zip(op, op_layers_names):
        if len(layer.shape) == 4:
            size = layer.shape[1]
            nof = layer.shape[3]
            display_layer = np.zeros((size, size * nof))
            for i in range(nof):
                img = layer[0, :, :, i]

                img -= img.mean()
                img /= img.std()
                img *= 64
                img += 128
                img = np.clip(img, 0, 255).astype('uint8')

                display_layer[:, (size * i): (size * i) + size] = img

            scale = 20. / nof
            plt.figure(figsize=(scale * nof, scale))
            plt.imshow(display_layer, aspect='auto', cmap='viridis')
            plt.grid(False)
            plt.title(name)
            plt.show()


# checkLayersWeLisualize()


def delete():
    cats_pics = [os.path.join(train_cats_images_dir, fname) for fname in train_cats_names[1001:11999]]
    dogs_pics = [os.path.join(train_dogs_images_dir, fname) for fname in train_dogs_names[1001:11999]]

    print(len(cats_pics))
    print(len(dogs_pics))

    for i, x in enumerate(dogs_pics):
        os.remove(x)
        print(i)

# delete()
