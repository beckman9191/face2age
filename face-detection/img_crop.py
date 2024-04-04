import cv2
import scipy.io as sio
import os
from centerface import CenterFace

def process_single_img(folder_name, img_name, padding, confindency):
    img = cv2.imread(folder_name + img_name)
    height, width, _ = img.shape
    landmarks = True

    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(img, height, width, confindency)
    #else:
        #dets = centerface(img, threshold)

    #print(len(dets))
    for idx, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        #cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)

        # Extract top-left and bottom-right coordinates
        x1, y1 = int(boxes[0]), int(boxes[1])
        x2, y2 = int(boxes[2]), int(boxes[3])

        # Calculate padding based on the dimensions of the crop rectangle and padding ratio
        padding_width = int((x2 - x1) * padding)
        padding_height = int((y2 - y1) * padding)

        # Apply padding, ensuring we don't go beyond the image boundaries
        x1_padded = max(x1 - padding_width, 0)
        y1_padded = max(y1 - padding_height, 0)
        x2_padded = min(x2 + padding_width, width)
        y2_padded = min(y2 + padding_height, height)

        cropped_image = img[y1_padded:y2_padded, x1_padded:x2_padded]
        #cropped_image = img[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
        cv2.imwrite('result/crop/' + img_name + '_' + str(idx) + '.jpg', cropped_image)

def generate_img_set(img_folder_name, padding, confindency):
    img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    img_names = []

    if not os.path.exists(img_folder_name):
        print(f"The folder {img_folder_name} does not exist.")
        return img_names

    for file in os.listdir(img_folder_name):
        if os.path.isfile(os.path.join(img_folder_name, file)) and any(
                file.lower().endswith(ext) for ext in img_extensions):
            img_names.append(file)

    for img_name in img_names:
        process_single_img(img_folder_name + '/', img_name, padding, confindency)


if __name__ == '__main__':
    if not os.path.exists('result/crop/'):
        os.makedirs('result/crop/')

    padding = 2
    confindency = 0.7

    #img_name = '000388.jpg'
    #process_single_img('', img_name, padding, confindency)

    img_folder_name = 'image_data'
    generate_img_set(img_folder_name, padding, confindency)
