import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# RNN Definition
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x, h1_0, h2_0):
        out, _ = self.lstm1(x, h1_0)
        out, _ = self.lstm2(out, h2_0)
        out = self.dense(out[:, -1, :])
        return out

    def init_lstm_state(self, batch_size, num_hiddens, device):
        return (torch.zeros((1, batch_size, num_hiddens), device=device),
                torch.zeros((1, batch_size, num_hiddens), device=device))

# ================================================================== #
#                         1. Input Pipeline                          #
# ================================================================== #

dataset = pd.read_csv('./data/NSE-Tata-Global-Beverages-Limited/NSE-Tata-Global-Beverages-Limited.csv')
dataset["Date"] = pd.to_datetime(dataset.Date, format="%Y-%m-%d")
dataset.index = dataset['Date']

# plt.figure(figsize=(16, 8))
# plt.plot(dataset.index, dataset['Close'], label='Close Price History')
# plt.plot(dataset['Open'], label='Open Price History')
# plt.legend()
# plt.show()

dataset.sort_index(ascending=True, axis=0, inplace=True)
new_dataset = dataset[['Close']]
# print(new_dataset.head())

final_dataset = new_dataset.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)
# print(scaled_data)
train_data = scaled_data[0:988, :]
valid_data = scaled_data[987:, :]
print(valid_data.shape)

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i])
    y_train_data.append(scaled_data[i])

x_test_data, y_test_data = [], []
for i in range(len(scaled_data)-len(valid_data), len(scaled_data)):
    x_test_data.append(scaled_data[i-60: i])
    y_test_data.append(scaled_data[i])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_test_data, y_test_data = np.array(x_test_data), np.array(y_test_data)
np.random.seed(7)
indices = np.arange(len(x_train_data))
np.random.shuffle(indices)
x_train_data = x_train_data[indices]
y_train_data = y_train_data[indices]

x_train = torch.from_numpy(x_train_data).float()
y_train = torch.from_numpy(y_train_data).float()
x_test = torch.from_numpy(x_test_data).float()
y_test = torch.from_numpy(y_test_data).float()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# ================================================================== #
#                         2. Initialization                          #
# ================================================================== #

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param_name, param in m.named_parameters():
            if "weight" in param_name:
                nn.init.xavier_uniform_(param)

model = Model(input_size, hidden_size, output_size)
model.apply(init_weights)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

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
        for i, (inputs, labels) in enumerate(train_loader):
            h1_0 = model.init_lstm_state(batch_size, hidden_size, device)
            h2_0 = model.init_lstm_state(batch_size, hidden_size, device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs, h1_0, h2_0)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i+1) % 5 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        loss_history.append(loss_sum / len(train_loader))
else:
    model = torch.load('StockPredict.ckpt')

# ================================================================== #
#                              4. Test                               #
# ================================================================== #

model.eval()
with torch.no_grad():
    h1_0 = model.init_lstm_state(len(x_test), hidden_size, device)
    h2_0 = model.init_lstm_state(len(x_test), hidden_size, device)
    output_pred = model(x_test.to(device), h1_0, h2_0).cpu()

# ================================================================== #
#                              5. Plot                               #
# ================================================================== #

output_pred = output_pred.numpy()
output_pred = scaler.inverse_transform(output_pred)
valid_data = scaler.inverse_transform(valid_data)
plt.figure(figsize=(16, 8))
plt.plot(output_pred, label='Predict')
plt.plot(valid_data, label='Raw')
plt.legend()
plt.show()

# ================================================================== #
#                              6. Save                               #
# ================================================================== #
if not disable_training:
    torch.save(model, 'StockPredict.ckpt')




