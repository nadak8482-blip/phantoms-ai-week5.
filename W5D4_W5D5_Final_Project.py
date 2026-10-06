import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DATA_DIR = "./data"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

class MNISTEnhancedClassifier(nn.Module):
    def __init__(self):
        super(MNISTEnhancedClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(), 
            nn.Dropout(p=0.25), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

model = MNISTEnhancedClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_accuracies = []

print("\n--- بدء التدريب مع الـ Dropout والـ ReLU ---")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
    accuracy = (correct / total) * 100
    test_accuracies.append(accuracy)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mnist_dropout_model.pth")
print("\n✨ تم حفظ النموذج المعدل بنجاح!")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', color='blue', linestyle='-')
plt.title('Training Loss with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), test_accuracies, marker='o', color='purple', linestyle='-')
plt.title('Test Accuracy with Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics_dropout.png')
plt.show()
print("✨ تم حفظ رسومات الأداء بنجاح!")
