import torch
from MyModels import myCNN  # Import the model class
from torch import nn
from train import test
import torchvision.models as models



# Create an instance of the model
CNNmodel = myCNN('MyCNN')

# Load the saved state dictionary
CNNmodel.load_state_dict(torch.load('result/trained_model_MyCNN.pth'))

# Set the model to evaluation mode
CNNmodel.eval()

if torch.backends.cuda.is_built():
  device = "cuda"
  print(f"Using GPU: {torch.cuda.get_device_name(device)}")
elif torch.has_mps:
  device = "mps"
else:
  device = "cpu"

risk_epoch, accuracy_epoch = test(CNNmodel, nn.MSELoss(), device)
print(f"test loss is {risk_epoch}")
print(f"MAE is {accuracy_epoch}")

# Create an instance of the model
resnet18 = models.resnet18(pretrained=True)
# Modify ResNet18
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 1)
# Load the saved state dictionary
resnet18.load_state_dict(torch.load('result/trained_model_resnet18.pth'))

# Set the model to evaluation mode
resnet18.eval()


risk_epoch, accuracy_epoch = test(resnet18, nn.MSELoss(), device)
print(f"test loss is {risk_epoch}")
print(f"MAE is {accuracy_epoch}")