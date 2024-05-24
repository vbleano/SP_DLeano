import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import absl.logging

def normalizeImage(image, label):
    return(tf.cast(image, tf.float32)/255.0, label)

def trainModel(train_data, validation_data):
    training_images = train_data.map(normalizeImage,
                                     num_parallel_calls = tf.data.AUTOTUNE)
    training_images = training_images.cache()
    training_images = training_images.prefetch(tf.data.AUTOTUNE)

    validation_images = validation_data.map(normalizeImage,
                                     num_parallel_calls = tf.data.AUTOTUNE)
    validation_images = validation_images.cache()
    validation_images = validation_images.prefetch(tf.data.AUTOTUNE)

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(255,255)))
    model.add(layers.Reshape((255,255,1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(20,activation='softmax'))
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    es = EarlyStopping(patience=5)
    model.fit(training_images,epochs=100,validation_data=validation_images,callbacks=[es])

    loss, accuracy = model.evaluate(training_images)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    absl.logging.set_verbosity(absl.logging.ERROR)
    model.save('image_classifier.model')