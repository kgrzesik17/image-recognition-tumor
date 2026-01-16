import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from keras.utils import image_dataset_from_directory
from keras import layers, Model
from keras.models import Sequential
from keras.utils import plot_model
from keras.metrics import CategoricalAccuracy, TruePositives, FalsePositives, FalseNegatives, Accuracy
import tensorflow_datasets as tfds

directory_train = 'BrainTumorDataset/train'
directory_test = 'BrainTumorDataset/test'

positive_train = os.path.join(directory_train, 'yes')
negative_train = os.path.join(directory_train, 'no')
positive_test = os.path.join(directory_test, 'yes')
negative_test = os.path.join(directory_test, 'no')

print('Zbiór uczący')
print(f'Liczba wyników pozytywnych {len(os.listdir(positive_train))}')
print(f'Liczba wyników negatywnych {len(os.listdir(negative_train))}')

print('Zbiór testowy')
print(f'Liczba wyników pozytywnych {len(os.listdir(positive_test))}')
print(f'Liczba wyników negatywnych {len(os.listdir(negative_test))}')

train_dataset, valid_dataset = image_dataset_from_directory(directory_train, validation_split=0.2,
                                                            subset='both',
                                                            seed=1410,
                                                            image_size=(225, 225),
                                                            label_mode='categorical',
                                                            color_mode="rgb")

test_dataset = image_dataset_from_directory(directory_test, seed=1410,
                                            image_size=(225, 225),
                                            label_mode='categorical',
                                            color_mode="rgb")

class_names = train_dataset.class_names
print(class_names)

fig, ax = plt.subplots(4, 4, figsize=(10,10))

for images, labels in train_dataset.take(1):
  for i in range(16):
    ax = plt.subplot(4,4,i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[int(np.argmax(labels[i]))])
    plt.axis("off")

# plt.show()

model = Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(225, 225, 3)),
    layers.Conv2D(16, 10, strides=2, padding='same', activation="relu"),
    layers.Conv2D(16, 8, padding='same', activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Conv2D(16, 7, padding='same', activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 7, padding='same', activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 5, padding='same', activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 5, padding='same', activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(256, 3, padding='same', activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(2, activation="softmax")
    ])

# plot_model(model, show_shapes=True)
model.summary()

# compiling
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])


# learning
history = model.fit(train_dataset, epochs=30, validation_data=valid_dataset,
                    batch_size=32)

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
plt.show()

result = model.evaluate(test_dataset, return_dict=True)

print(f'Dokładność modelu {round(result["accuracy"]*100,2)}%')