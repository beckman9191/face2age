from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras.preprocessing.image import img_to_array, load_img, save_img

# Define your data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotation degrees
    width_shift_range=0.2,   # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Random horizontal flip
    fill_mode='nearest'      # Fill mode for new pixels
)



def augment_images(input_dir, output_dir, target_count):
    # List all images in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    # Calculate the number of augmentations needed per image
    augmentations_per_image = max(target_count // len(image_files), 1)

    # Loop over each image file
    for image_file in image_files:
        img_path = os.path.join(input_dir, image_file)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(img_array, batch_size=1,
                                  save_to_dir=output_dir,
                                  save_prefix='aug',
                                  save_format='jpg'):
            i += 1
            if i >= augmentations_per_image:
                break  # Stop after generating enough augmentations

# Define your input and output directories
age_groups = range(50, 90)
counter = 0
for age in age_groups:
    input_dir_train = f'dataset_new/age/train/{age}'
    output_dir_train = input_dir_train  # Augment in-place
    augment_images(input_dir_train, output_dir_train, target_count=500)
    print('--(' + str(counter) + ')Processing--')
    counter += 1


    input_dir_test = f'dataset_new/age/test/{age}'
    output_dir_test = input_dir_test  # Augment in-place
    augment_images(input_dir_test, output_dir_test, target_count=100)
