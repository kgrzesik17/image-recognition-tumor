import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from keras.utils import image_dataset_from_directory
from keras import layers, Model
from keras.models import Sequential
from keras.metrics import CategoricalAccuracy, Recall
import tensorflow_datasets as tfds

directory_train = 'BrainTumorDataset/train'
directory_test = 'BrainTumorDataset/test'

positive_train = os.path.join(directory_train, 'yes')
negative_train = os.path.join(directory_train, 'no')
positive_test = os.path.join(directory_test, 'yes')
negative_test = os.path.join(directory_test, 'no')

print('Training Set')
count_yes_train = len(os.listdir(positive_train))
count_no_train = len(os.listdir(negative_train))
print(f'Positive results: {count_yes_train}')
print(f'Negative results: {count_no_train}')

print('Test Set')
print(f'Positive results: {len(os.listdir(positive_test))}')
print(f'Negative results: {len(os.listdir(negative_test))}')

# mertic selection
total_train = count_yes_train + count_no_train
max_ratio = max(count_yes_train, count_no_train) / total_train
print(f"Dominant class ratio: {round(max_ratio * 100, 2)}%")

# use recall if dominant class > 60%
if max_ratio <= 0.6:
    print("Decision: Data is balanced -> Metric: ACCURACY")
    selected_metrics = ['accuracy']
    metric_key = 'accuracy'
else:
    print("Decision: Data is imbalanced -> Metric: RECALL")
    # We track Recall as the main metric, but keep accuracy for reference
    selected_metrics = [Recall(name='recall'), 'accuracy']
    metric_key = 'recall'

# loading data
train_dataset, valid_dataset = image_dataset_from_directory(directory_train, validation_split=0.2,
                                                            subset='both',
                                                            seed=1410,
                                                            image_size=(256, 256),
                                                            label_mode='categorical',
                                                            color_mode="rgb")

test_dataset = image_dataset_from_directory(directory_test, seed=1410,
                                            image_size=(256, 256),
                                            label_mode='categorical',
                                            color_mode="rgb")

class_names = train_dataset.class_names
print(f"Classes: {class_names}")

fig, ax = plt.subplots(4, 4, figsize=(10,10))
for images, labels in train_dataset.take(1):
  for i in range(16):
    ax = plt.subplot(4,4,i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[int(np.argmax(labels[i]))])
    plt.axis("off")
plt.show()


def create_model_shallow():
    # architecture 1 - shallow: fast training
    return Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(256, 256, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ], name="Model_1_Shallow")

def create_model_standard():
    # architecture 2 - standard - deeper, batch normalization
    return Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(256, 256, 3)),
        layers.Conv2D(16, 10, strides=2, padding='same', activation="relu"),
        layers.Conv2D(16, 8, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 7, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 5, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax")
    ], name="Model_2_Standard")

def create_model_deep_gap():
    # architecture 3: deep with global average pooling
    return Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(256, 256, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(), 
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ], name="Model_3_Deep_GAP")

models_to_run = [create_model_shallow, create_model_standard, create_model_deep_gap]


for model_builder in models_to_run:
    model = model_builder()
    print(f"\n{'='*50}")
    print(f"STARTING EXPERIMENT: {model.name}")
    print(f"{'='*50}")
    model.summary()

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=selected_metrics)

    history = model.fit(train_dataset, epochs=30, validation_data=valid_dataset, batch_size=32, verbose=1)

    plt.figure(figsize=(8, 6))
    plt.plot(history.history[metric_key], label=f'Train {metric_key}')
    plt.plot(history.history[f'val_{metric_key}'], label=f'Val {metric_key}')
    
    plt.title(f'Learning Curve: {model.name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_key.capitalize())
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print(f"\nTEST RESULTS FOR {model.name}:")
    result = model.evaluate(test_dataset, return_dict=True)
    
    if metric_key == 'recall':
        print(f"-> RECALL: {round(result['recall']*100, 2)}%")
        if 'accuracy' in result:
             print(f"-> Accuracy (auxiliary): {round(result['accuracy']*100, 2)}%")
    else:
        print(f"-> ACCURACY: {round(result['accuracy']*100, 2)}%")
    
    print("-" * 50)