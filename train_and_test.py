from CNNmodel import myCNN
from load_data import test_loader, train_loader, test_dataset, train_dataset
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time



batch_size = 100
def test(model, loss_function, device):
    # we first move our model to the configured device
    model = model.to(device = device)

    # we make sure we are not tracking gradient
    # gradient is used in training, we do not need it for test
    with torch.no_grad():
        total_loss = 0
        total_mae = 0  # Total Mean Absolute Error
        total_samples = 0

        # loop over test mini-batches
        for i, (images, labels) in enumerate(test_loader):
            # reshape labels to have the same form as output
            # make sure labels are of torch.float32 type
            labels = labels.view(-1, 1).float()


            # move tensors to the configured device
            images = images.to(device = device)

            labels = labels.to(device = device)


            # forward pass
            outputs = model(images)

            loss = loss_function(outputs, labels)

            # Calculate the absolute difference between predicted and actual ages
            absolute_difference = torch.abs(outputs - labels)

            # Compute the average absolute difference
            average_difference = absolute_difference.mean().item()

            # Update total loss and MAE
            total_loss += loss.item()
            total_mae += absolute_difference.sum().item()
            total_samples += labels.size(0)

        # average test risk and accuracy over the whole test dataset
        test_loss = total_loss / len(test_loader)
        average_mae = total_mae / total_samples


    return test_loss, average_mae


expected_train_batches = len(train_dataset) // batch_size + (len(train_dataset) % batch_size != 0)
expected_test_batches = len(test_dataset) // batch_size + (len(test_dataset) % batch_size != 0)

actual_train_batches = len(train_loader)
actual_test_batches = len(test_loader)

print(f"Expected number of training batches: {expected_train_batches}")
print(f"Actual number of training batches: {actual_train_batches}")
print(f"Expected number of test batches: {expected_test_batches}")
print(f"Actual number of test batches: {actual_test_batches}")

model = myCNN()
if torch.backends.cuda.is_built():
  device = "cuda"
  print(f"Using GPU: {torch.cuda.get_device_name(device)}")
elif torch.has_mps:
  device = "mps"
else:
  device = "cpu"

#test_risk, test_acc = test(model, nn.MSELoss(), device)
#(f"test risk is {test_risk}")
#print(f"test accuracy is {test_acc}")

def train(model, num_epochs, device):
    # we first move our model to the configured device
    model = model.to(device = device)

    # set loss to binary CE
    loss_function = nn.MSELoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Initiate the values
    train_risk = []
    test_risk = []
    test_mae = []

    for epoch in range(num_epochs):
        # training risk in one epoch
        risk = 0
        start_time = time.time()
        counter = 0
        # loop over training data
        for i, (images, labels) in enumerate(train_loader):

            # reshape labels to have the same form as output
            # make sure labels are of torch.float32 type
            labels = labels.view(-1, 1).float()

            # move tensors to the configured device
            images = images.to(device = device)

            labels = labels.to(device = device)


            # forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # collect the training loss
            risk += loss.item()

            # backward pass
            optimizer.zero_grad()
            # complete: compute the gradient of loss
            # use auto-grad (just 1 line)
            loss.backward()
            # one step of gradient descent
            optimizer.step()
            counter += 1
            print(counter)


        # test out model after update by the optimizer
        risk_epoch, accuracy_epoch = test(model, loss_function, device)

        # collect losses and accuracy
        train_risk.append(risk/i)
        test_risk.append(risk_epoch)
        test_mae.append(accuracy_epoch)

        # we can print a message every second epoch
        # Start timing


        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_risk[-1]:.4f}, Test Loss: {test_risk[-1]:.4f}, Test MAE: {test_mae[-1]:.4f}')

        end_time = time.time()
        # Calculate the time difference in seconds
        epoch_time = end_time - start_time
        print(f"Time spent for these 2 epoch: {epoch_time} seconds")

    # plot the losses
    plt.plot([i+1 for i in range(num_epochs)], train_risk, label='training risk')
    plt.plot([i+1 for i in range(num_epochs)], test_risk, label='test risk')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()

    # plot the accuracy
    plt.plot([i+1 for i in range(num_epochs)], test_mae, label='Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.show()

    return train_risk, train_risk, test_accuracy

train_risk, train_risk, test_accuracy = train(model, 20, device)


save_path = 'trained_model.pth'
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')