from MyModels import myCNN
from load_data import test_loader, train_loader, test_dataset, train_dataset
import torch
from train import train, batch_size
import numpy as np



#------------------------------------------------------------------------------#
expected_train_batches = len(train_dataset) // batch_size + (len(train_dataset) % batch_size != 0)
expected_test_batches = len(test_dataset) // batch_size + (len(test_dataset) % batch_size != 0)

actual_train_batches = len(train_loader)
actual_test_batches = len(test_loader)

print(f"Expected number of training batches: {expected_train_batches}")
print(f"Actual number of training batches: {actual_train_batches}")
print(f"Expected number of test batches: {expected_test_batches}")
print(f"Actual number of test batches: {actual_test_batches}")

model = myCNN("MyCNN")
if torch.backends.cuda.is_built():
  device = "cuda"
  print(f"Using GPU: {torch.cuda.get_device_name(device)}")
elif torch.has_mps:
  device = "mps"
else:
  device = "cpu"

train_risk, test_risk, test_mae = train(model, 40 , device, 'myCNN')

with open('result/risk_{}.txt'.format('myCNN'), 'w') as f:
    f.write('train risk: \n')
    matrix = np.array(train_risk)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')

    f.write('test risk: \n')
    matrix = np.array(test_risk)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')

    f.write('test MAE: \n')
    matrix = np.array(test_mae)
    np.savetxt(f, matrix, fmt='%f')
    f.write('\n')
