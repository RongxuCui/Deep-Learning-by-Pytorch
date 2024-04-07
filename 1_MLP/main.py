import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
input_size = 28 * 28
hidden_size = 256
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MLP Definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output


# ================================================================== #
#                         1. Input Pipeline                          #
# ================================================================== #

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# ================================================================== #
#                         2. Initialization                        #
# ================================================================== #

model = MLP(input_size, hidden_size, num_classes)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# loss function
criterion = nn.CrossEntropyLoss()

# ================================================================== #
#                             3. Train                               #
# ================================================================== #
disable_training = False
if not disable_training:
    total_step = len(train_loader)
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        for i, (images, labels) in enumerate(train_loader):
            inputs = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        loss_history.append(loss_sum / len(train_loader))
else:
    model = torch.load('MLP.ckpt')

# ================================================================== #
#                              4. Test                               #
# ================================================================== #

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        inputs = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy on the test dataset: {} %'.format(100 * correct / total))

# ================================================================== #
#                              5. Plot                               #
# ================================================================== #
if not disable_training:
    plt.plot(range(num_epochs), loss_history)
    plt.show()

# ================================================================== #
#                              5. Save                               #
# ================================================================== #
if not disable_training:
    torch.save(model, 'MLP.ckpt')