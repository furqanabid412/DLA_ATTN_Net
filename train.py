import os
from datetime import datetime
from dataloader.dataloader import PL_DataLoader
from trainer import PLModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.pl_utils import TQDM_Bar


def train(cfg,is_pretrained=False):

    dataset = PL_DataLoader(train_dir=cfg['dataset']['root']['train'],
                           val_dir=cfg['dataset']['root']['val'],
                           test_dir=None,
                           batch_size=cfg['train']['batch_size'],
                           num_workers=cfg['train']['workers'],
                           shuffle=cfg['train']['shuffle'])
    dataset.setup('fit')


    if is_pretrained:
        model = PLModel.load_from_checkpoint(cfg['train']['pretrained_dir'], config=cfg, verbose=False)
    else:
        model = PLModel(config=cfg, verbose=False)

    # bar = pl.callbacks.TQDMProgressBar()
    bar = TQDM_Bar()
    ckp_name = str(datetime.now().year) + '_' + str(datetime.now().month) + '_' + str(datetime.now().day) \
               + '_' + str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second)
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg['train']['log_dir']),verbose=True,every_n_epochs=1, save_last=True,
                                              filename=ckp_name + '_{epoch}_{val_miou:.6f}'+'_pretrained={}'.format(is_pretrained), monitor='val_miou',mode='max',save_top_k=2)
    if cfg['train']['logging']:
        wandb = WandbLogger(project=cfg['train']['wandb_log_proj'])
        trainer = pl.Trainer(gpus=cfg['train']['gpus'],auto_scale_batch_size=False,enable_checkpointing=True,precision=32,
                                 logger=wandb,callbacks=[checkpoint, bar], max_epochs=cfg['train']['max_epochs'],strategy=cfg['train']['strategy'])
    else:
        trainer = pl.Trainer(gpus=cfg['train']['gpus'],auto_scale_batch_size=False,enable_checkpointing=True,precision=32,
                                 logger=False,callbacks=[bar], max_epochs=cfg['train']['max_epochs'],strategy=cfg['train']['strategy'])

    trainer.fit(model, dataset)


def validate(cfg):
    dataset = PL_DataLoader(train_dir=cfg['dataset']['root']['train'],
                           val_dir=cfg['dataset']['root']['val'],
                           test_dir=None,
                           batch_size=cfg['validate']['batch_size'],
                           num_workers=cfg['validate']['workers'],
                           shuffle=cfg['validate']['shuffle'])
    dataset.setup('fit')

    model = PLModel.load_from_checkpoint(cfg['validate']['pretrained_dir'], config=cfg, verbose=False)

    bar = pl.callbacks.TQDMProgressBar()

    if cfg['validate']['logging']:
        wandb = WandbLogger(project=cfg['validate']['wandb_log_proj'])
        trainer = pl.Trainer(gpus=cfg['validate']['gpus'], auto_scale_batch_size=False, enable_checkpointing=False,
                             precision=32, logger=wandb, callbacks=[bar], max_epochs=1, strategy=cfg['validate']['strategy'])
    else:
        trainer = pl.Trainer(gpus=cfg['validate']['gpus'], auto_scale_batch_size=False, enable_checkpointing=False,
                             precision=32, logger=False, callbacks=[bar], max_epochs=1,strategy=cfg['validate']['strategy'])

    trainer.validate(model,dataset)