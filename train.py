import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from model import UNet
from transforms import ToTensorMultiChannel

def train_model(data_dir, batch_size=1, num_epochs=10, learning_rate=0.001):
    # Create dataset and dataloader with custom transformation
    custom_transform = ToTensorMultiChannel()
    dataset = BraTSDataset(data_dir=data_dir, transform=custom_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Save trained model
    model_dir = '/Users/udisharora/Desktop/brAT/model'   # Specify your model directory
    os.makedirs(model_dir, exist_ok=True)
    model_filename = 'my_model.pth'
    model_path = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully at: {model_path}")

    return model

