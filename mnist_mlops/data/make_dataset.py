import torch
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms


image_names = os.listdir()

data_folder = "data/raw/"
files = os.listdir("data/raw/")

train_images = []
test_images = []

train_targets = []
test_targets = []


transform_normalize = transforms.Normalize(mean=1, std=1)
for i in files:
    if "train_images" in i:
        train_images.append(transform_normalize(torch.load(data_folder + i)))
        train_targets.append(torch.load(data_folder + i.replace("images", "target")))

    elif "test_images" in i:
        test_images.append(transform_normalize(torch.load(data_folder + i)))
        test_targets.append(torch.load(data_folder + i.replace("images", "target")))
train_images, train_targets, test_images, test_targets = (
    torch.concatenate(train_images),
    torch.concatenate(train_targets),
    torch.concatenate(test_images),
    torch.concatenate(test_targets),
)
# Create a TensorDataset
train_images = train_images.unsqueeze(1)
dataset_train = TensorDataset(train_images, train_targets)

# Creating the validation set :
generator = torch.Generator().manual_seed(42)
dataset_train, dataset_validation = random_split(dataset_train, [0.75, 0.25], generator)
# Specify batch size for DataLoader
batch_size = 64
# Create a DataLoader for train set
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
# Create DataLoader for validation set
val_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)

# Same for test images :
test_images = test_images.unsqueeze(1)
dataset_test = TensorDataset(test_images, test_targets)
test_loader = DataLoader(dataset_test)
results = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}

torch.save(results, data_folder.replace("raw", "processed") + "processed_tensor.pt")