from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator object
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_gen = datagen.flow_from_directory(
    directory='path_to_your_images',
    target_size=(256, 256),
    classes=['normal_road', 'mildly_bad_road', 'severely_bad_road'],
    class_mode='categorical',
    batch_size=32,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    directory='path_to_your_images',
    target_size=(256, 256),
    classes=['normal_road', 'mildly_bad_road', 'severely_bad_road'],
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)

# Instantiate the model
model = Sequential()

# Add the first Convolutional layer
model.add(Conv2D(64, (3,3), input_shape=(256, 256, 3), activation='relu'))

# Add the second Convolutional layer
model.add(Conv2D(64, (3,3), activation='relu'))

# Add the Max Pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten the tensor output from the previous layer
model.add(Flatten())

# Add the fully connected layer with 3 neurons (for 3 classes)
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=20)

# Evaluate the model
score = model.evaluate(val_gen)
print('Test accuracy:', score[1])
