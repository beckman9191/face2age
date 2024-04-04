import cv2
import os
import torch
from centerface import CenterFace
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def process_single_img(folder_name, img_name, padding, confindency):
    img = cv2.imread(folder_name + img_name)
    height, width, _ = img.shape
    landmarks = True

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('models/resnet18.pth', map_location=torch.device('cpu')))

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(img, height, width, confindency)

    age_list = []
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
        x1 = max(x1 - padding_width, 0)
        y1 = max(y1 - padding_height, 0)
        x2 = min(x2 + padding_width, width)
        y2 = min(y2 + padding_height, height)

        cropped_image = img[y1:y2, x1:x2]

        image_pil = Image.fromarray(cropped_image).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0)

        with torch.no_grad():  # No need to track gradients
            output = model(image_tensor)
        age = int(output[0][0])
        age_list.append(age)

        #cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (2, 255, 0), 1)

        if y1 > 15:
            text_position = ((x1 + x2) // 2 - 10, y1 - 10)  # Adjust the 10 pixels offset as needed
        else:
            text_position = ((x1 + x2) // 2, y2 + 10)

        # Define the font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        #font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 0.7
        font_color = (50, 255, 0)  # White color
        line_type = 2

        # Draw the text (e.g., the score or index number) above the square
        cv2.putText(img, str(age), text_position, font, font_scale, font_color, line_type)

    cv2.imwrite('result/' + img_name, img)

    return age_list

def draw_and_save_age_distribution_pie_chart(age_list, filename):
    # Define age intervals and labels
    intervals = [(0, 26), (26, 36), (36, 46), (46, 56), (56, 66), (66, 76), (76, 86), (86, float('inf'))]
    interval_labels = ['0-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', 'Over 85']

    # Count ages in each interval
    age_counts = {label: 0 for label in interval_labels}
    for age in age_list:
        for (start, end), label in zip(intervals, interval_labels):
            if start <= age < end:
                age_counts[label] += 1
                break

    # Remove intervals with 0 count
    age_counts = {k: v for k, v in age_counts.items() if v > 0}

    # Define a color palette with bright and light colors around yellow and orange
    colors = ['#ffcc00', '#ffdd55', '#ffee88', '#ffffbb', '#ffaa00',
              '#ffbb33', '#ffcc66', '#ffdd99', '#ffeecc']

    # Create pie chart
    plt.figure(figsize=(8, 7))
    plt.pie(age_counts.values(), labels=age_counts.keys(), autopct='%1.1f%%', colors=colors, startangle=140)

    # Place title at the bottom of the chart
    plt.figtext(0.5, 0.08, 'Age Distribution', ha='center', va='bottom', fontsize=12)

    # Save the chart to a file
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory


if __name__ == '__main__':
    if not os.path.exists('result/'):
        os.makedirs('result/')

    padding = 0.5
    confindency = 0.7

    img_name = '3.jpg'
    age_list = process_single_img('', img_name, padding, confindency)

    draw_and_save_age_distribution_pie_chart(age_list, 'result/age_distribution_chart.png')
