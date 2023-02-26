import torch
import warnings
import os
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import yaml
from dataloader.semanticKitti import SemanticKitti
from dataloader.NuscRange import NuscRange
import matplotlib.pyplot as plt
import numpy as np
from trainer import PLModel

def visualize_range_image(y, title, image_name):
    my_dpi = 600
    factor = 32
    w = (960 / my_dpi) * factor
    h = (32 / my_dpi) * factor

    fig, ax = plt.subplots(layout='constrained')
    ax.set_axis_off()
    fig.set_size_inches(w, h, forward=True)
    # ax.set_title(title)
    nusc_config = yaml.safe_load(open('configs/config.yaml', 'r'))
    label_colormap = nusc_config['dataset']['color_map']
    colors = np.array([[label_colormap[val] for val in row] for row in y], dtype='B')
    ax.imshow(colors)
    ax.figure.savefig(image_name, dpi=my_dpi)


if __name__ == '__main__':
    frame_number = 50

    config = yaml.safe_load(open('configs/kitti_config.yaml', 'r'))
    dataset = SemanticKitti(cfg=config)

    # max_values = []
    # for i in range(len(dataset)):
    #     frame = dataset[i]
    #     x1 = frame['projected_rangeimg']
    #     x1 = x1[0].max()
    #     max_values.append(x1)
    #
    # max_values = np.array(max_values)

    # config = yaml.safe_load(open('configs/kitti_config.yaml', 'r'))
    # dataset = SemanticKitti(cfg=config)
    frame = dataset[frame_number]
    x = frame['projected_rangeimg']
    y = frame['projected_labels']
    x = torch.from_numpy(x)
    # x = x.numpy()
    # x = x[0]

    x = x[None, :, :, :]
    img_means = torch.FloatTensor([12.1063, 7.7884, -0.3015, -0.3165, -0.7672])
    img_stds = torch.FloatTensor([19.4655, 12.2054, 9.5711, 10.7408, 1.5764])

    # img_means = torch.FloatTensor([12.1063,12.12, 10.88, 0.23, -1.04])
    # img_stds = torch.FloatTensor([19.4655,12.32, 6.91, 0.86, 0.16])
    # img_means = torch.FloatTensor([12.12, 0.21, 10.88, 0.23, -1.04])
    # img_stds = torch.FloatTensor([12.32, 0.16, 6.91, 0.86, 0.16])


    x = (x - img_means[None, :, None, None]) / img_stds[None, :, None, None]

    model_config = yaml.safe_load(open('configs/config.yaml', 'r'))
    model_baseline = PLModel.load_from_checkpoint(model_config['train']['pretrained_dir'], config=model_config, verbose=False)
    logits = model_baseline.forward(x)
    preds = torch.argmax(logits, 1)

    gt = torch.from_numpy(y)
    accu = preds[0] == gt
    accu = accu.sum()/32768
    print(accu)
    # save_path = 'D:\iros2023/results/'
    visualize_range_image(preds[0].numpy(), title='',image_name='kitti_' + str(frame_number) + '.png')
    visualize_range_image(y, title='', image_name='gt_' + str(frame_number) + '.png')
