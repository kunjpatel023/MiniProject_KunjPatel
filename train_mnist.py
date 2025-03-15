import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ssl
import certifi

# Fix SSL verification
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context

# Define lightweight CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # 1 input channel (grayscale), 16 filters
        self.pool = nn.MaxPool2d(2, 2)    # Pooling layer
        self.conv2 = nn.Conv2d(16, 32, 3) # 16 input channels, 32 filters
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # Fully connected layer
        self.fc2 = nn.Linear(120, 10)     # Output layer (10 digits)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and train the model
model = SimpleCNN()

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):  # 3 epochs
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

# Save the state dictionary
torch.save(model.state_dict(), "mnist_simplecnn.pth")
print("Model saved as mnist_simplecnn.pth")