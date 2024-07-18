import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as sk
from sklearn.preprocessing import StandardScaler
import random
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('FullWeatherData_Toronto.csv')

# Extract input columns and output columns
input_cols = df.columns[:-4]
output_cols = [c for c in df.columns if c.isdigit()]

# Convert the dataframe to a numpy array
timeseries = df[input_cols].values.astype('float32')

print(timeseries.shape)

# Normalize the data
scaler = StandardScaler()
timeseries = scaler.fit_transform(timeseries)

# Train-test split for time series
train_size = int(len(timeseries) * 0.6)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

# Function to create dataset for multi-time step prediction
def create_dataset(dataset, lookback, n_steps, total_features, output_features):
    X, y = [], []
    total_features_list = list(total_features)
    output_indices = [total_features_list.index(feature) for feature in output_features]
    for i in range(len(dataset) - lookback - n_steps + 1):
        X.append(dataset[i:i + lookback, :])
        y.append(dataset[i + lookback:i + lookback + n_steps, output_indices])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

# Parameters for dataset creation
lookback = 8
n_steps = 24  # Number of future time steps to predict

# Create training and test datasets
X_train, y_train = create_dataset(train, lookback=lookback, n_steps=n_steps, total_features=input_cols, output_features=output_cols)
X_test, y_test = create_dataset(test, lookback=lookback, n_steps=n_steps, total_features=input_cols, output_features=output_cols)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return outputs, hidden, cell

# Define the Attention class
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim, requires_grad=True))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention, dim=1)

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.lstm = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat((input, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        prediction = self.relu(prediction)  # Applying ReLU to ensure non-negative outputs
        return prediction, hidden, cell

# Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0, :]

        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t, :] if teacher_force else output

        return outputs

# Define dimensions
INPUT_DIM = len(input_cols)  # Replace with actual input dimension
OUTPUT_DIM = len(output_cols)  # Replace with actual output dimension
ENC_HID_DIM = 50
DEC_HID_DIM = 50
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# Instantiate models
enc = Encoder(INPUT_DIM, ENC_HID_DIM, n_layers=2, dropout=ENC_DROPOUT)
attn = Attention(ENC_HID_DIM)
dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, n_layers=2, dropout=DEC_DROPOUT, attention=attn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = Seq2Seq(enc, dec, device).to(device)

# Initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Assuming X_train, y_train, X_test, y_test are already defined as tensors
train_data = data.TensorDataset(X_train, y_train)
train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)

# Move training and test data to device once
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Lists to store metrics over epochs
train_rmse_list, train_mae_list, train_r2_list = [], [], []
test_rmse_list, test_mae_list, test_r2_list = [], [], []

# Training loop
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch, y_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # Validation every few epochs
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train, y_train)
            train_loss = loss_fn(y_pred_train, y_train)
            train_rmse = torch.sqrt(train_loss)
            train_mae = torch.mean(torch.abs(y_pred_train - y_train))
            train_r2 = sk.r2_score(y_train.cpu().numpy().flatten(), y_pred_train.cpu().numpy().flatten())

            y_pred_test = model(X_test, y_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_rmse = torch.sqrt(test_loss)
            test_mae = torch.mean(torch.abs(y_pred_test - y_test))
            test_r2 = sk.r2_score(y_test.cpu().numpy().flatten(), y_pred_test.cpu().numpy().flatten())

            # Append metrics to lists
            train_rmse_list.append(train_rmse.item())
            train_mae_list.append(train_mae.item())
            train_r2_list.append(train_r2)
            test_rmse_list.append(test_rmse.item())
            test_mae_list.append(test_mae.item())
            test_r2_list.append(test_r2)

            print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, MAE {train_mae:.4f}, R^2 {train_r2:.4f} | Test RMSE {test_rmse:.4f}, MAE {test_mae:.4f}, R^2 {test_r2:.4f}")

# Plotting Metrics Over Epochs
epochs = range(0, n_epochs, 2)

plt.figure(figsize=(15, 5))
plt.plot(epochs, train_rmse_list, label='Train RMSE')
plt.plot(epochs, test_rmse_list, label='Test RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('RMSE Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(epochs, train_mae_list, label='Train MAE')
plt.plot(epochs, test_mae_list, label='Test MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('MAE Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(epochs, train_r2_list, label='Train R²')
plt.plot(epochs, test_r2_list, label='Test R²')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.title('R² Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

def plot_forecast_comparison_plotly(timeseries, y_pred_test, node_indices, train_size, lookback, n_steps, output_cols, selected_indices=None):
    if selected_indices is None:
        selected_indices = node_indices  # If selected_indices is None, plot all node_indices

    for node_idx in selected_indices:
        col_idx = output_cols.index(str(node_idx))

        # Initialize figure
        fig = go.Figure()

        # Historical data (first 60% of the dataset)
        historical_data_len = train_size
        historical_data_x = list(range(-historical_data_len, 0)) # Convert range to list
        historical_data_y = timeseries[:historical_data_len, col_idx]
        fig.add_trace(go.Scatter(x=historical_data_x, y=historical_data_y, mode='lines', name='Historical Data'))

        # Actual data for the last 40% of the dataset
        actual_data_len = len(timeseries) - historical_data_len
        actual_data_x = list(range(0, actual_data_len)) # Convert range to list
        actual_data_y = timeseries[historical_data_len:, col_idx]
        fig.add_trace(go.Scatter(x=actual_data_x, y=actual_data_y, mode='lines', name='Actual Data'))

        # Forecasted data for the last 40% (Test Predictions)
        forecast_data_y = np.full(actual_data_len, np.nan)
        for j in range(y_pred_test.shape[0]):
            start_idx = lookback + j
            end_idx = start_idx + n_steps
            forecast_data_y[start_idx:end_idx] = y_pred_test[j, :, col_idx]


        forecast_data_x = list(range(0, actual_data_len)) # Convert range to list


        fig.add_trace(go.Scatter(x=forecast_data_x, y=forecast_data_y, mode='lines', name='Forecast'))

        # Update layout
        fig.update_layout(title=f'Forecast Comparison for Node {node_idx}',
                          xaxis_title='Number of Samples (Time Steps)',
                          yaxis_title='Values',
                          legend=dict(x=0, y=1, traceorder='normal'),
                          autosize=False,
                          width=2500,
                          height=700)

        # Show the plot
        fig.show()

# Plotting Predictions vs. Actual for Nodes 7684 and 8018
with torch.no_grad():
    y_pred_train = model(X_train.to(device), y_train.to(device)).cpu().numpy()
    y_pred_test = model(X_test.to(device), y_test.to(device)).cpu().numpy()

    node_indices = [7684, 8018]

    for node_idx in node_indices:
        col_idx = output_cols.index(str(node_idx))

        plt.figure(figsize=(30, 6))
        plt.plot(timeseries[:, col_idx], label='Actual')  # Plot the actual data for the current output variable

        # Shift train predictions for plotting
        train_plot = np.ones_like(timeseries[:, col_idx]) * np.nan
        for j in range(y_pred_train.shape[0]):
            train_plot[lookback + j:lookback + j + n_steps] = y_pred_train[j, :, col_idx]
        plt.plot(train_plot, c='r', label='Train Predictions')

        # Shift test predictions for plotting
        test_plot = np.ones_like(timeseries[:, col_idx]) * np.nan
        for j in range(y_pred_test.shape[0]):
            test_plot[train_size + lookback + j:train_size + lookback + j + n_steps] = y_pred_test[j, :, col_idx]
        plt.plot(test_plot, c='g', label='Test Predictions')

        plt.title(f"Node {node_idx}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    node_indices = [7684, 8018]
    plot_forecast_comparison_plotly(timeseries, y_pred_test, node_indices, train_size, lookback, n_steps, output_cols, selected_indices=node_indices)
