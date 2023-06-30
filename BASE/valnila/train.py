import lightning.pytorch as pl
from dataset import MinstDataModule
from model import LitAutoEncoder

# init the autoencoder
autoencoder = LitAutoEncoder()


logger = pl.loggers.TensorBoardLogger("runs", name="my_model")
dataset = MinstDataModule("data/MINST", batch_size=32)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=100, logger=logger)
trainer.fit(model=autoencoder, train_dataloaders=dataset)