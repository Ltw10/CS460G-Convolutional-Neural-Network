import torch
import torch.nn as nn
import torch.optim as optim
from dataset import read_train_sets, load_train, DataSet

# Define the convolutional neural network architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=1046)
        self.fc2 = nn.Linear(in_features=1046, out_features=num_classes)

    # Input tensor is supposed to be (batch_size, num_channels, height, width)
    # From debugging appears to be (batch_size, height, width, num_channels)
    # 32, 32, 32, 3        
    def forward(self, x):
        #Comes in as (batch_size, height, width, num_channels)?
        # Transpose dimensions to (batch_size, num_channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Apply convolutions
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        # Flatten and apply fully connected layers 32 X 8 X 8
        x = x.reshape(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
def main():
    # Define the hyperparameters
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.01

    # Load the train and validation DataSets
    datasets = read_train_sets("./data/training_data", 32, ['pembroke', 'cardigan'], 0.1)
    train_data = datasets.train
    validation_data = datasets.valid

    # Create the CNN
    cnn = CNN(num_classes=train_data.num_examples())

    # Define the loss function and optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Create a PyTorch data loader objects for the training and validation datasets
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data.images()), torch.from_numpy(train_data.labels()))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(validation_data.images()), torch.from_numpy(validation_data.labels()))
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

    # Train the neural network
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            # Forward pass
            outputs = cnn(images)

            # Convert one-hot labels to binary labels - grabs the index of the 1
            labels = torch.max(labels, 1)[1]

            loss = lossFunction(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2 == 0:
                print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

        # Validate the neural network
        accuracy = test_cnn(cnn, validation_loader)
        print("Epoch [{}/{}], Valdation Accuracy: {} %".format(epoch+1, num_epochs, accuracy))

        cnn.train()

    # Save the model
    torch.save(cnn.state_dict(), "./cnn_model2.pt")

    test_data_images, test_data_labels, test_data_img_names, test_data_cls = load_train("./data/testing_data", 32, ['pembroke', 'cardigan'])
    test_data = DataSet(test_data_images, test_data_labels, test_data_img_names, test_data_cls)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data.images()), torch.from_numpy(test_data.labels()))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    accuracy = test_cnn(cnn, test_loader)
    print("Test Accuracy: {} %".format(accuracy))


def test_cnn(cnn, dataloader):
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = torch.max(labels,1)[1]
            correct += (predicted == labels).sum().item()

        accuracy = 100 * (correct / total)
        return accuracy
    
def test_on_saved_cnn():
    num_pictures = 267
    test_data_images, test_data_labels, test_data_img_names, test_data_cls = load_train("./data/testing_data", 32, ['pembroke', 'cardigan'])
    test_data = DataSet(test_data_images, test_data_labels, test_data_img_names, test_data_cls)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data.images()), torch.from_numpy(test_data.labels()))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    model = CNN(num_pictures)
    model.load_state_dict(torch.load("./cnn_model.pt"))
    
    accuracy = test_cnn(model, test_loader)
    print("Test Accuracy using Saved Model: {} %".format(accuracy))

test_on_saved_cnn()
# main()