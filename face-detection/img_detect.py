import cv2
import scipy.io as sio
import os
from centerface import CenterFace

def process_single_img(folder_name, img_name):
    img = cv2.imread(folder_name + img_name)
    h, w = img.shape[:2]
    landmarks = True

    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(img, h, w, threshold=0.35)
    else:
        dets = centerface(img, threshold=0.35)

    #print(len(dets))
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)

    '''
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(img, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    '''
    cv2.imwrite('result/detect/' + img_name + '.jpg', img)

def generate_img_set(img_folder_name):
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
        process_single_img(img_folder_name + '/', img_name)


if __name__ == '__main__':
    if not os.path.exists('result/detect/'):
        os.makedirs('result/detect/')

    img_name = '000388.jpg'
    process_single_img('', img_name)

    #img_folder_name = 'image_data'
    #generate_img_set(img_folder_name)
