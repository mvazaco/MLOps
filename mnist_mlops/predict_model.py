import torch
import numpy as np
from models.model import SimpleCNN
import os
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import click


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--model", help="name of the folder where the checkpoint is found.")
@click.option("--new_data", help="path to the new data to predict. ")
@click.option("--test_name", help="Name of the prediction (for subfolder generation).")
def make_predictions(model, new_data, test_name):
    # Read environment variables if provided, otherwise use default values

    model = os.getenv("MODEL", model)
    new_data = os.getenv("NEW_DATA", new_data)
    test_name = os.getenv("TEST_NAME", test_name)

    # Loading and processing new data in Numpy format :

    data_files = os.listdir(new_data)
    transform_normalize = transforms.Normalize(mean=1, std=1)
    new_images = []
    for i in data_files:
        if "npy" in i:
            data = transform_normalize(torch.from_numpy(np.load(new_data + i)))
            new_images.append(data)

    new_images = torch.concatenate(new_images)
    new_images = new_images.unsqueeze(1)
    dataset_new = TensorDataset(new_images)
    new_loader = DataLoader(dataset_new)

    # Loading the pre-trained model :
    model = SimpleCNN()
    model.load_state_dict(torch.load(model))
    predictions = predict(model, new_loader)

    # Saving the predictions :

    torch.save(predictions, "tests/predictions_{}.pt".format(test_name))


cli.add_command(make_predictions)

if __name__ == "__main__":
    cli()