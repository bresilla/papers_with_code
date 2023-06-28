import pytorch_lightning as pl
import config
from model import PixModel
from dataset import DataModule


def main():
    datamoudle = DataModule()
    model = PixModel()
    logger = pl.loggers.TensorBoardLogger('logs/', name='pix2pix')
    trainer = pl.Trainer(logger=logger, max_epochs=config.num_epochs)
    trainer.fit(model, datamoudle)
        
if __name__ == '__main__':
    main()