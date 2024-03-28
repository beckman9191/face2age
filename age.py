# Importing dependencies
import pandas as pd
import numpy as np
import cv2
import os
import csv
from sklearn.model_selection import train_test_split


# the size of the image
width = 224
height = 224
# Loading dataset
meta = pd.read_csv('meta.csv')
train_csv = 'train.csv'
test_csv = 'test.csv'
# Dropping gender column
meta = meta.drop(['gender'], axis=1)

# Filtaring dataset
meta = meta[meta['age'] >= 0]
meta = meta[meta['age'] <= 101]

# Converting into numpy array
meta = meta.values

# Spliting dataset into training and testing set
D_train, D_test = train_test_split(meta, test_size=0.2, random_state=42)
age_train = [row[0] for row in D_train]
age_test = [row[0] for row in D_test]
original_path_train = [row[1] for row in D_train]
original_path_test = [row[1] for row in D_test]

# Making the directory structure
for i in range(102):
    output_dir_train_male = 'dataset/age/train/' + str(i)
    output_dir_train_female = 'dataset/age/train/' + str(i)

    if not os.path.exists(output_dir_train_male):
        os.makedirs(output_dir_train_male)

    if not os.path.exists(output_dir_train_female):
        os.makedirs(output_dir_train_female)

    output_dir_test_male = 'dataset/age/test/' + str(i)
    output_dir_test_female = 'dataset/age/test/' + str(i)

    if not os.path.exists(output_dir_test_male):
        os.makedirs(output_dir_test_male)

    if not os.path.exists(output_dir_test_female):
        os.makedirs(output_dir_test_female)

# Finally making the training and testing set
counter = 0

with open(train_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)



    for image in D_train:
        img = cv2.imread(image[1], 1)
        img = cv2.resize(img, (height ,width))
        path = 'dataset/age/train/' + str(image[0]) + '/' + str(counter) + '.jpg'
        cv2.imwrite(path, img)
        writer.writerow([original_path_train[counter], path, age_train[counter]])
        print('--('+str(counter)+')Processing--')
        counter += 1

print(f'CSV file "{train_csv}" created successfully.')
counter = 0

with open(test_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for image in D_test:
        img = cv2.imread(image[1], 1)
        img = cv2.resize(img, (height ,width))
        path = 'dataset/age/train/' + str(image[0]) + '/' + str(counter) + '.jpg'

        cv2.imwrite(path, img)
        writer.writerow([original_path_test[counter], path, age_train[counter]])
        print('--('+str(counter)+')Processing--')
        counter += 1

print(f'CSV file "{test_csv}" created successfully.')


