import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def predict_model():
    # Preprocessing
    training_dir = os.path.join("data", "train")
    image_size = (55, 94, 3)
    target_size = (55, 94)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.5
    )
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=target_size,
        class_mode='binary'
    )

    # CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=image_size),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile and Train Model
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = EarlyStopping(monitor='accuracy', patience=5)

    model.fit(
        train_generator,
        epochs=30,
        callbacks=[early_stop]
    )

    return model

if __name__ == '__main__':
    model = predict_model()
    model.save("model_predict.h5")
