import pytorch_lightning as pl
from dataloader.NUSC import NuscenseDataset
from torch.utils.data import DataLoader

class PL_DataLoader(pl.LightningDataModule):
    def __init__(self,cfg, batch_size=4,num_workers=2,shuffle=True,drop_last=True):
        super(PL_DataLoader, self).__init__()

        self.config = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = NuscenseDataset(cfg=self.config,stage='train')
            self.val_dataset = NuscenseDataset(cfg=self.config,stage='val')
        if stage == 'test':
            pass

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,batch_size=self.batch_size,
                                     shuffle=self.shuffle,num_workers=self.num_workers,
                                     pin_memory=False,drop_last=self.drop_last)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                      pin_memory=False, drop_last=self.drop_last)
        return dataloader

    def test_dataloader(self):
        pass
        # dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
        #                               shuffle=self.shuffle, num_workers=self.num_workers,
        #                               pin_memory=False, drop_last=self.drop_last)
        # return dataloader