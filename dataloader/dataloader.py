import pytorch_lightning as pl
from .NuscRange import NuscRange
from torch.utils.data import DataLoader

class PL_DataLoader(pl.LightningDataModule):
    def __init__(self,train_dir=None,val_dir=None,test_dir=None, batch_size=4,num_workers=2,shuffle=True,drop_last=True):
        super(PL_DataLoader, self).__init__()

        self.train_root = train_dir
        self.val_root = val_dir
        self.test_root = test_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = NuscRange(root=self.train_root)
            self.val_dataset = NuscRange(root=self.val_root)
        if stage == 'test':
            self.test_dataset = NuscRange(root=self.test_root)

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
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                      pin_memory=False, drop_last=self.drop_last)
        return dataloader