from MyModels import myCNN
from load_data import test_loader, train_loader, test_dataset, train_dataset
import torch
from train import train, batch_size
import numpy as np
import os


save_dir = 'result_augmented'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#------------------------------------------------------------------------------#
expected_train_batches = len(train_dataset) // batch_size + (len(train_dataset) % batch_size != 0)
expected_test_batches = len(test_dataset) // batch_size + (len(test_dataset) % batch_size != 0)

actual_train_batches = len(train_loader)
actual_test_batches = len(test_loader)

print(f"Expected number of training batches: {expected_train_batches}")
print(f"Actual number of training batches: {actual_train_batches}")
print(f"Expected number of test batches: {expected_test_batches}")
print(f"Actual number of test batches: {actual_test_batches}")

cnn = myCNN()
cnn_name = 'MyCNN'

if torch.backends.cuda.is_built():
  device = "cuda"
  print(f"Using GPU: {torch.cuda.get_device_name(device)}")
elif torch.has_mps:
  device = "mps"
else:
  device = "cpu"

train_risk, test_risk, train_mae, test_mae = train(cnn, 20 , device, 'myCNN', save_dir)

save_path = f'{cnn_name}.pth'
torch.save(cnn.state_dict(), save_path)
print(f'Model saved to {save_path}')

with open(f'{save_dir}/risk_{cnn_name}.txt', 'w') as f:
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
