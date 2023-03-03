import warnings
warnings.filterwarnings('ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import yaml
from dataloader.semanticKitti import SemanticKitti
from dataloader.NuscRange import NuscRange
from dataloader.NUSC import NuscenseDataset
import matplotlib.pyplot as plt
import numpy as np
from trainer import PLModel
from tqdm import tqdm
from dataloader.dataloaderv2 import PL_DataLoader

# def visualize_range_image(y, title, image_name):
#     my_dpi = 600
#     factor = 32
#     w = (960 / my_dpi) * factor
#     h = (32 / my_dpi) * factor
#
#     fig, ax = plt.subplots(layout='constrained')
#     ax.set_axis_off()
#     fig.set_size_inches(w, h, forward=True)
#     # ax.set_title(title)
#     nusc_config = yaml.safe_load(open('configs/config.yaml', 'r'))
#     label_colormap = nusc_config['dataset']['color_map']
#     colors = np.array([[label_colormap[val] for val in row] for row in y], dtype='B')
#     ax.imshow(colors)
#     ax.figure.savefig(image_name, dpi=my_dpi)
#     plt.close(fig)

if __name__ == '__main__':

    cfg = yaml.safe_load(open('configs/config.yaml', 'r'))
    dataset_config = yaml.safe_load(open('configs/nusc_config.yaml', 'r'))
    dataset = PL_DataLoader(cfg=dataset_config, batch_size=cfg['validate']['batch_size'], num_workers=cfg['validate']['workers'], shuffle=cfg['validate']['shuffle'])
    dataset.setup('fit')
    model = PLModel.load_from_checkpoint(cfg['validate']['pretrained_dir'], config=cfg, verbose=False)
    bar = pl.callbacks.TQDMProgressBar()

    if cfg['validate']['logging']:
        wandb = WandbLogger(project=cfg['validate']['wandb_log_proj'])
        trainer = pl.Trainer(gpus=cfg['validate']['gpus'], auto_scale_batch_size=False, enable_checkpointing=False,
                             precision=32, logger=wandb, callbacks=[bar], max_epochs=1,
                             strategy=cfg['validate']['strategy'])
    else:
        trainer = pl.Trainer(gpus=cfg['validate']['gpus'], auto_scale_batch_size=False, enable_checkpointing=False,
                             precision=32, logger=False, callbacks=[bar], max_epochs=1,
                             strategy=cfg['validate']['strategy'])

    trainer.validate(model, dataset)

    # config = yaml.safe_load(open('configs/nusc_config.yaml', 'r'))
    # img_means = torch.FloatTensor([12.1063, 7.7884, -0.3015, -0.3165, -0.7672])
    # img_stds = torch.FloatTensor([19.4655, 12.2054, 9.5711, 10.7408, 1.5764])
    # model_config = yaml.safe_load(open('configs/config.yaml', 'r'))
    # model_baseline = PLModel.load_from_checkpoint(model_config['train']['pretrained_dir'], config=model_config, verbose=False)
    # dataset = NuscenseDataset(cfg=config)
    # save_path = 'E:\source_code\saved/'
    #
    # for frame_number in tqdm(range(len(dataset))):
    #     frame = dataset[frame_number]
    #     x = frame['projected_rangeimg']
    #     y = frame['projected_labels']
    #     x = torch.from_numpy(x)
    #     x = x[None, :, :, :]
    #     x = (x - img_means[None, :, None, None]) / img_stds[None, :, None, None]
    #     logits = model_baseline.forward(x)
    #     preds = torch.argmax(logits, 1)
    #     gt = torch.from_numpy(y)
    #     accu = preds[0] == gt
    #     accu = accu.sum()/32768
    #     # print(accu)
    #     img_name = 'kitti_' + str(frame_number)+('_')+str(accu.item()) + '.png'
    #     visualize_range_image(preds[0].numpy(), title='',image_name=os.path.join(save_path,img_name))
    #     visualize_range_image(y, title='', image_name=os.path.join(save_path,'gt_' + str(frame_number) + '.png'))