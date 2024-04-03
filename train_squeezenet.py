from load_data import test_loader, train_loader, test_dataset, train_dataset
import torch.nn as nn
import torch
import torchvision.models as models
import os
import numpy as np
from train import batch_size, train

save_dir = 'result_augmented'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

expected_train_batches = len(train_dataset) // batch_size + (len(train_dataset) % batch_size != 0)
expected_test_batches = len(test_dataset) // batch_size + (len(test_dataset) % batch_size != 0)

actual_train_batches = len(train_loader)
actual_test_batches = len(test_loader)

print(f"Expected number of training batches: {expected_train_batches}")
print(f"Actual number of training batches: {actual_train_batches}")
print(f"Expected number of test batches: {expected_test_batches}")
print(f"Actual number of test batches: {actual_test_batches}")

if torch.backends.cuda.is_built():
  device = "cuda"
  print(f"Using GPU: {torch.cuda.get_device_name(device)}")
elif torch.has_mps:
  device = "mps"
else:
  device = "cpu"



# two pretrained models
squeezenet = models.squeezenet1_0(pretrained=True)

# Modify SqueezeNet
squeezenet.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, 1, kernel_size=(1, 1)),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1))
)
squeezenet.num_classes = 1


# -------------------------------- start training -------------------------------- #
epoch_num = 20


train_risk, test_risk, train_mae, test_mae = train(squeezenet, epoch_num, device, "squeezenet", save_dir)

save_path = 'squeezenet.pth'
torch.save(squeezenet.state_dict(), save_path)
print(f'Model saved to {save_path}')

with open(f'{save_dir}/risk_squeezenet.txt', 'w') as f:
    f.write('train risk: \n')
    matrix = np.array(train_risk)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')

    f.write('test risk: \n')
    matrix = np.array(test_risk)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')

    f.write('train MAE: \n')
    matrix = np.array(train_mae)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')

    f.write('test MAE: \n')
    matrix = np.array(test_mae)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')