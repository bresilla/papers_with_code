import lightning.pytorch as pl
from dataset import DataModule
from model import CycleGAN
import config

# init the autoencoder
autoencoder = CycleGAN()


logger = pl.loggers.TensorBoardLogger("runs", name="CycleGAN")
dataset = DataModule(data_root=config.DATA_ROOT, batch_size=config.BATCH_SIZE)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=100, logger=logger)
trainer.fit(model=autoencoder, train_dataloaders=dataset)