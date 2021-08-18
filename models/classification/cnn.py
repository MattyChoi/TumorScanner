from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense


def cnn(input_shape=(150, 150, 3)):
    # outputs 3d feature maps (height, width, features)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # coverts 3d features to 1d vectors
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

model = cnn()

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

batch_size = 16

# augmentation configuration used for training; rescaling images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip = True)

# augmentation configuration for testing: only rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)

# generator that will read pictures found and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'Augmented/',
        target_size = (150, 150),
        batch_size = batch_size,
        class_mode= 'binary')

validation_generator = test_datagen.flow_from_directory(
        'MRI_Scans/', # target directory
        target_size=(150, 150), # all images will be resized
        batch_size=batch_size,
        class_mode='binary') # binary labels

model.fit(train_generator,
        epochs=50,
        batch_size=16,
        validation_data=validation_generator)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save("model.h5")
print("Saved model to disk")