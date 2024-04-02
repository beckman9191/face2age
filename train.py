import time
from load_data import test_loader, train_loader, test_dataset, train_dataset
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

batch_size = 64
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



def train(model, num_epochs, device, model_name):
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
            if counter % 50 == 0:
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
        print(f"Time spent for this epoch: {epoch_time} seconds")

    save_dir = 'result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plot the losses
    plt.figure()
    plt.plot([i + 1 for i in range(num_epochs)], train_risk, label='training risk')
    plt.plot([i + 1 for i in range(num_epochs)], test_risk, label='test risk')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig('result/Loss_{}.png'.format(model_name))

    # plot the accuracy
    plt.figure()
    plt.plot([i + 1 for i in range(num_epochs)], test_mae, label='Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.savefig("result/MAE_{}.png".format(model_name))

    save_path = "result/trained_model_{}.pth".format(model_name)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return train_risk, test_risk, test_mae
