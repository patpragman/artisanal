import numpy as np
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model
from sklearn.metrics import classification_report
import yaml
from pprint import pprint
from pathlib import Path
from math import sqrt

HOME_DIRECTORY = Path.home()
SEED = 42

def calculate_conv_layer_output_size(n, p, f, s):
    return int((n + 2*p - f)/s + 1)

class ArtisanalCNN(nn.Module):
    """
    Artisanal CNN with a few tunable hyperparameters

    """
    def __init__(self,
                 image_size,
                 layer_1_filters, layer_2_filters, layer_3_filters,  # hyperparameters
                 num_classes, activation_function="relu"):
        super(ArtisanalCNN, self).__init__()

        if activation_function.lower() == "relu":
            fn = nn.ReLU()
        elif activation_function.lower() == "leaky_relu":
            fn = nn.LeakyReLU(0.1)
        else:
            fn = nn.Tanh()

        # calculate the size of the layers
        self.layer_1_output_size = calculate_conv_layer_output_size(sqrt(image_size/3), 1, 4, 2)//2  # divided by 2 for the pooling
        self.layer_2_output_size = calculate_conv_layer_output_size(self.layer_1_output_size, 1, 4, 2)//2
        self.layer_3_output_size = calculate_conv_layer_output_size(self.layer_2_output_size, 1, 4, 2)  # no division because no pool

        print(self.layer_1_output_size)
        print(self.layer_2_output_size)
        print(self.layer_3_output_size)

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=layer_1_filters,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(layer_1_filters),
            fn,
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=layer_1_filters,
                      out_channels=layer_2_filters,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(layer_2_filters),
            fn,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=layer_2_filters, out_channels=layer_3_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_3_filters),
            fn)
        self.fully_connected_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(layer_3_filters*self.layer_3_output_size**2, 1024),
            fn)
        self.fully_connected_2 = nn.Sequential(
            nn.Linear(1024, num_classes))



    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        # flatten out here
        x = x.reshape(x.size(0), -1)
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)

        return x


with open("cnn_sweep.yml", "r") as yaml_file:
    sweep_config = yaml.safe_load(yaml_file)

sweep_id = wandb.sweep(sweep=sweep_config)
def find_best_model():
    # config for wandb

    # Initialize wandb
    wandb.init(project="Artisanal CNN")
    config = wandb.config

    # creating the model stuff
    input_size = 3*config.input_size**2
    num_classes = 2  # this doesn't ever change
    learning_rate = config.learning_rate
    epochs = wandb.config.epochs

    filter_size = config.filter_sizes

    print('HYPER PARAMETERS:')
    pprint(config)
    # Create the CNN-based image classifier model
    model = ArtisanalCNN(input_size,
                         filter_size, filter_size, filter_size,
                         num_classes, activation_function=config.activation_function)

    print('Model Architecture:')
    print(model)

    path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"

    dataset = FloatImageDataset(directory_path=path,
                                true_folder_name="entangled", false_folder_name="not_entangled"
                                )

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
    batch_size = config.batch_size

    # create the dataloaders
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimzer parsing logic:
    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
                                   device="cpu", wandb=wandb, verbose=False)

    y_true, y_pred = history['y_true'], history['y_pred']
    print(classification_report(y_true=y_true, y_pred=y_pred))

    # Log test accuracy to wandb

    # Log hyperparameters to wandb
    wandb.log(dict(config))

if __name__ == "__main__":


    wandb.agent(sweep_id, function=find_best_model)

    # Specify your W&B project and sweep ID
    project_name = "Artisanal CNN"

    # Fetch sweep runs
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    runs = list(sweep.runs)

    # Find the best run based on the metric you care about (e.g., lowest validation loss)
    best_run = None
    best_metric_value = float("inf")

    for run in runs:
        if run.summary["accuracy"] > best_metric_value:
            best_run = run
            best_metric_value = run.summary["accuracy"]

    # Print the best run and its hyperparameters
    print("Best Run:")
    print(f"Run ID: {best_run.id}")
    print(f"Test Accuracy: {best_run.summary['accuracy']}")
    print(f"Hyperparameters: {best_run.config}")

    with open("artisanal_results.md", "w") as write_file:
        write_file.writelines(
            ["Best Run:", f"Run ID: {best_run.id}",
             f"Test Accuracy: {best_run.summary['accuracy']}",
             f"Hyperparameters: {best_run.config}"
             ]
        )
