import lightning.pytorch as pl
from dataset import DataModule as MyData
from model import PixModel as MyModel
import config

logger = pl.loggers.TensorBoardLogger('runs', name='pix2pix')
dataset = MyData()
trainer = pl.Trainer(logger=logger, max_epochs=config.num_epochs)
model = MyModel()
trainer.fit(model=model, train_dataloaders=dataset)