import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import cv2

# Create an instance of the model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Load the saved state dictionary
#model.load_state_dict(torch.load('models/resnet18.pth'))
model.load_state_dict(torch.load('models/resnet18.pth', map_location=torch.device('cpu')))


# Set the model to evaluation mode
model.eval()

# Load an image and apply transformations
#image = Image.open('1.jpg').convert('RGB')

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = cv2.imread('1.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image).convert('RGB')
image_tensor = transform(image_pil).unsqueeze(0)  # Add a batch dimension

# Make a prediction
with torch.no_grad():  # No need to track gradients
    output = model(image_tensor)

print(output)
print(int(output[0][0]))

