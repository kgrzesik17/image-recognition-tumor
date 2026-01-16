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