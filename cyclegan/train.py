import lightning.pytorch as pl
from dataset import DataModule
from model2 import CycleGAN
import config

# init the autoencoder
autoencoder = CycleGAN(learning_rate=config.LEARNING_RATE)
logger = pl.loggers.TensorBoardLogger("runs", name="CycleGAN")
dataset = DataModule(data_root=config.DATA_ROOT, batch_size=config.BATCH_SIZE)
trainer = pl.Trainer(limit_train_batches=16, max_epochs=200, logger=logger)
trainer.fit(model=autoencoder, train_dataloaders=dataset)