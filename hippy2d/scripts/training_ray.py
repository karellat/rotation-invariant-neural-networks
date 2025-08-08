import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.loggers import CSVLogger

# 1) Define your LightningModule, consuming a `config` dict
class LitMNIST(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, config["hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_size"], 10)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

# 2) Data loaders factory
def get_data_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    train, val = random_split(mnist, [55000, 5000])
    return (
        DataLoader(train, batch_size=batch_size, num_workers=1, shuffle=True),
        DataLoader(val, batch_size=batch_size, num_workers=1),
    )

# 3) Training function that Ray Tune will call
def train_tune(config):
    # Model + data
    model = LitMNIST(config)
    train_loader, val_loader = get_data_loaders(config["batch_size"])

    # Logger and callback to report metrics back to Tune
    logger = CSVLogger(save_dir="logs", name="tune_pl")
    tune_callback = TuneReportCallback(
        {"val_loss": "val_loss", "val_acc": "val_acc"},
        on="validation_end"
    )

    # Each Trainer here will use 1 GPU
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[tune_callback],
        enable_progress_bar=False
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    # 4) Define the search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "hidden_size": tune.choice([64, 128, 256]),
    }

    # 5) Scheduler & reporter
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "hidden_size"],
        metric_columns=["val_loss", "val_acc", "training_iteration"]
    )

    # 6) Launch tuning: allocate 1 GPU per trial → with 2 GPUs available, you get 2 concurrent trials
    analysis = tune.run(
        train_tune,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="mnist_tuning",
    )

    print("Best hyperparameters found were: ", analysis.get_best_config(
        metric="val_loss", mode="min"
    ))
