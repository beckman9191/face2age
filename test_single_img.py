import torch
from CNNmodel import myCNN  # Import the model class
from torchvision import transforms
from PIL import Image

# Create an instance of the model
model = myCNN()

# Load the saved state dictionary
model.load_state_dict(torch.load('result/trained_model.pth'))

# Set the model to evaluation mode
model.eval()

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Example transformation
    transforms.ToTensor(),
    # Add other transformations as needed
])

# Load an image and apply transformations
image = Image.open('./d77.png').convert('RGB')
#image = Image.open('./dataset/age/test/27/461.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

# Make a prediction
with torch.no_grad():  # No need to track gradients
    output = model(image_tensor)

print(output)

