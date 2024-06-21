import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the model structure (should match the trained model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 3)  # 2 classes: cat and dog
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load('plant_disease.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class_dict = {
    0: "healthy",
    1: "early",
    2: "late"
}

# Function to predict the class of an image
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        class_name = class_dict[class_idx]
    
    return class_name


for i in range(1, 16):
    
    # Example usage
    image_path = f'./test/{i}.JPG'  # Replace with your image path
    print(f'The image is a {predict_image(image_path)}.')