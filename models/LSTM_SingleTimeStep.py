import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as sk

df = pd.read_csv('FullWeatherData_Toronto.csv')

input_cols = df.columns[:-4]
output_cols = []

for c in df.columns:
    if c.isdigit():
        output_cols.append(c)

timeseries = df[input_cols].values.astype('float32')  # extract all columns excluding time

print(timeseries.shape)

# train-test split for time series
train_size = int(len(timeseries) * 0.6)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

# # Plot the rainfall data for each output column
# for col in output_cols:
#     plt.figure(figsize=(10, 6))
#     plt.plot(df[col], label=col)
#     plt.title("Rainfall Data for {}".format(col))
#     plt.xlabel("Time")
#     plt.ylabel("Rainfall")
#     plt.legend()
#     plt.show()

def create_dataset(dataset, lookback, total_features, output_features):
    X, y = [], []
    total_features_list = list(total_features)
    output_indices = [total_features_list.index(feature) for feature in output_features]
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i + lookback, :])
        y.append(dataset[i + 1:i + lookback + 1, output_indices[0]:output_indices[-1] + 1])
    return torch.tensor(X), torch.tensor(y)


lookback = 10

X_train, y_train = create_dataset(train, lookback=lookback, total_features=input_cols, output_features=output_cols)
X_test, y_test = create_dataset(test, lookback=lookback, total_features=input_cols, output_features=output_cols)

print("X_train shape:", str(X_train.shape))

class AirModel(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz, dropout_prob=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_sz, hidden_size=hidden_sz, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(hidden_sz, output_sz)

    def forward(self, x):
        x, _ = self.lstm(x)
        #x = self.dropout(x)  # Apply dropout regularization
        x = self.linear(x)
        x = torch.relu(x)
        return x

input_size = len(input_cols)
output_size = len(output_cols)
model = AirModel(input_size, hidden_sz=50, output_sz=output_size, dropout_prob=0.1)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=32)

print("output size:", str(output_size))
print("Train samples", str(train_size))

# Training loop
n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_loss = loss_fn(y_pred_train, y_train)
            train_rmse = np.sqrt(train_loss)
            train_mae = torch.mean(torch.abs(y_pred_train - y_train))
            train_r2 = sk.r2_score(y_train.numpy().flatten(), y_pred_train.numpy().flatten())

            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_rmse = np.sqrt(test_loss)
            test_mae = torch.mean(torch.abs(y_pred_test - y_test))
            test_r2 = sk.r2_score(y_test.numpy().flatten(), y_pred_test.numpy().flatten())

            print("Epoch %d: Train RMSE %.4f, MAE %.4f, R^2 %.4f | Test RMSE %.4f, MAE %.4f, R^2 %.4f" % (
            epoch, train_rmse, train_mae, train_r2, test_rmse, test_mae, test_r2))

# Plotting
with torch.no_grad():
    # Extract predictions
    y_pred_train = model(X_train)[:, -1, :].numpy()
    y_pred_test = model(X_test)[:, -1, :].numpy()

    # Plot both train and test predictions on the same plot for each output variable
    for i, col in enumerate(output_cols):
        plt.figure(figsize=(30, 6))
        plt.plot(timeseries[:, i], label='Actual')  # Plot the actual data for the current output variable

        # Shift train predictions for plotting
        train_plot = np.ones_like(timeseries[:, i]) * np.nan
        print(len(train_plot))
        train_plot[lookback:train_size] = y_pred_train[:, i]  # Adjusted the index
        plt.plot(train_plot, c='r', label='Train Predictions')

        # Shift test predictions for plotting
        test_plot = np.ones_like(timeseries[:, i]) * np.nan
        test_plot[train_size+lookback:len(timeseries)] = y_pred_test[:, i]  # Adjusted the index
        plt.plot(test_plot, c='g', label='Test Predictions')

        plt.title(col)
        plt.legend()
        plt.tight_layout()
        plt.show()



