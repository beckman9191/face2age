from age_dataset import AgeDataset
from torch.utils.data import DataLoader
import torchvision

batch_size = 100

train_dataset = AgeDataset('dataset/age/train')
print(train_dataset.__getitem__(2))
print(train_dataset.__getitem__(2)[0].shape)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = AgeDataset('dataset/age/test')
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

for i, (images, labels) in enumerate(train_loader):
    if i == 3:  # Display the first 3 batches
        break
    print(f"Batch {i + 1}")
    # Process the images and labels