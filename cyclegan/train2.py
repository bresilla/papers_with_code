import lightning.pytorch as pl
from dataset import DataModule as MyData
from model import CycleGAN_2 as MyModel
import config


model = MyModel(learning_rate=config.learning_rate)
logger = pl.loggers.TensorBoardLogger("runs", name="CycleGAN")
dataset = MyData(data_root=config.data_root, batch_size=config.batch_size)
trainer = pl.Trainer(limit_train_batches=16, max_epochs=200, logger=logger)
trainer.fit(model=model, train_dataloaders=dataset)