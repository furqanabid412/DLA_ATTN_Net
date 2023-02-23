import pytorch_lightning as pl

class TQDM_Bar(pl.callbacks.TQDMProgressBar):
    def on_validation_start(self, trainer, pl_module):
        pass
    def on_validation_batch_start( self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.has_dataloader_changed(dataloader_idx):
            return