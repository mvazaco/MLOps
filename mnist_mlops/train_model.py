import torch

from models.model import SimpleCNN
from torch import optim
import matplotlib.pyplot as plt
import os
import hydra
import logging

log = logging.getLogger(__name__)


@hydra.main(config_name="config_train.yaml")
def train(config):
    # Get the original working directory
    original_wd = hydra.utils.get_original_cwd()

    # Change back to the original working directory
    os.chdir(original_wd)

    lr = float(config["learning_rate"])
    training_name = config["training_name"]

    # Instantiate the model
    model = SimpleCNN()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    log.info("THE CURRENT DIRECTORY IS " + os.getcwd())
    data = torch.load("data/processed/processed_tensor.pt")
    train_loader = data["train_loader"]
    # Training loop
    num_epochs = 5
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_list.append(average_loss)
        log.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")
    plt.plot(loss_list)
    if not os.path.isdir("reports/figures/{}".format(training_name)):
        os.system("mkdir reports/figures/{}".format(training_name))
    plt.savefig("reports/figures/{}/training.png".format(training_name))
    if not os.path.isdir("models/{}".format(training_name)):
        os.system("mkdir models/{}".format(training_name))
    torch.save(model.state_dict(), "models/{}/checkpoint.pth".format(training_name))


if __name__ == "__main__":
    train()