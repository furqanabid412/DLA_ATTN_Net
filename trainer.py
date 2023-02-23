import torch
import pytorch_lightning as pl
from network.DLA_ATTN_NET import DLA_ATTN_NET
from losses.WASL import WAsymmetricLoss
from utils.metrics_eval import IouEval
from utils.knn import KNN
import torch.nn as nn


class PLModel(pl.LightningModule):
    def __init__(self, config = None,verbose = False):
        super(PLModel, self).__init__()

        self.verbose = verbose
        self.class_names = config['dataset']['class_names']
        self.n_classes = len(self.class_names)
        self.img_means, self.img_stds = torch.FloatTensor(config['dataset']['mean']), torch.FloatTensor(config['dataset']['std'])
        self.model = DLA_ATTN_NET(num_inputs=config['dataset']['n_channels'], channels=[64, 128, 128],num_outputs=self.n_classes)

        # loss
        epsilon_w = config['dataset']['epsilon']
        class_wise_pts = torch.tensor(config['dataset']['class_counts'])
        log_classweights = torch.log((class_wise_pts + epsilon_w))
        overall_wt = 1 / log_classweights
        self.criterion = WAsymmetricLoss(class_weights=1+overall_wt,gamma_neg=1.5,gamma_pos=1)
        # self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.evaluator = IouEval(n_classes=self.n_classes, device="cuda", ignore=config['dataset']['ignore_class'])
        self.val_evaluator = IouEval(n_classes=self.n_classes, device="cuda", ignore=config['dataset']['ignore_class'])

        # post-processing - for validation set only
        self.do_post_process = config['post_process']['apply']
        if self.do_post_process:
            self.postProcessing = KNN(knn=config['post_process']['knn'], search=config['post_process']['search'], sigma=config['post_process']['sigma'],
                                  cutoff=config['post_process']['cutoff'], nclasses=self.n_classes,apply_gauss=config['post_process']['IsGauss'])

        self.train_mean_iou,self.val_mean_iou = 0.0,0.0

    def forward(self, x):
        out = self.model(x)  # x : [B,C,H,W]  --> out : [B,n,H,W]
        return out

    def loss(self, y_hat, y):
        loss = self.criterion(y_hat,y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch["projected_rangeimg"], batch["projected_labels"]

        BS, _, H, W = x.shape
        self.img_means, self.img_stds = self.img_means.to(x.get_device()), self.img_stds.to(x.get_device())
        x = (x - self.img_means[None, :, None, None]) / self.img_stds[None, :, None, None]
        y = y.long()
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, 1)

        self.evaluator.addBatch(preds, y)
        iou_mean, class_iou = self.evaluator.getIoU()

        for i, iou in enumerate(class_iou):
            self.log('train-{}-iou'.format(self.class_names[i]), iou, on_step=True, on_epoch=False)

        self.log('train_step_miou', iou_mean, on_step=True, on_epoch=False, prog_bar= True)
        self.log('train_loss', loss, prog_bar= True)
        self.train_mean_iou = iou_mean
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_epoch_avg_miou', self.train_mean_iou, on_step=False, on_epoch=True,sync_dist=False)
        self.train_mean_iou = 0.0
        self.evaluator.reset()


    def validation_step(self, batch, batch_idx):
        x, y = batch["projected_rangeimg"], batch["projected_labels"]
        self.img_means, self.img_stds = self.img_means.to(x.get_device()), self.img_stds.to(x.get_device())
        x = (x - self.img_means[None, :, None, None]) / self.img_stds[None, :, None, None]
        y = y.long()
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, 1)

        if self.do_post_process:
            labels, batched_labels_pred, batched_labels_knn = self.post_processing(batch, preds)
            self.val_evaluator.addBatch(batched_labels_knn.type(torch.int64), labels.type(torch.int64))
        else :
            self.val_evaluator.addBatch(preds, y)

        iou_mean, class_iou = self.val_evaluator.getIoU()
        for i, iou in enumerate(class_iou):
            self.log('val-{}-iou'.format(self.class_names[i]), iou, on_step=True, on_epoch=False)

        self.log('val_iou', iou_mean, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False) # for multi-gpu training sync_dist = True otherwise False
        self.val_mean_iou = iou_mean


    def validation_epoch_end(self, outputs):
        self.log('val_miou', self.val_mean_iou, on_step=False, on_epoch=True, sync_dist=False)
        self.val_mean_iou = 0.0
        self.val_evaluator.reset()

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, test_step_outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]


    def post_processing(self,batch,preds):
        # getting other parameters for re-projection
        proj_depth, unproj_range, proj_x, proj_y = batch["proj_depth"], batch["unproj_range"], batch["proj_x"], batch["proj_y"]
        points, labels, mask = batch["points"], batch["labels"], batch["mask"]
        # post processing cannot be done in batches
        # so breaking the batches into single input
        bs, np, _ = points.shape
        batched_labels_pred = torch.zeros((bs, np)).cuda()
        batched_labels_knn = torch.zeros((bs, np)).cuda()
        for ii in range(bs):
            batched_labels_pred[ii] = preds[ii][proj_y[ii].long(), proj_x[ii].long()]
            if preds.is_cuda:
                pr_depth, un_range, pred = proj_depth[ii], unproj_range[ii], preds[ii]
                px, py = proj_x[ii], proj_y[ii]
            else:
                pr_depth, un_range, pred = proj_depth[ii].cpu(), unproj_range[ii].cpu(), preds[ii]
                px, py = proj_x[ii].cpu(), proj_y[ii].cpu()

            un_range_ = un_range[mask[ii]]
            px_ = px[mask[ii]]
            py_ = py[mask[ii]]
            pts = list(un_range_.shape)

            batched_labels_knn[ii, :pts[0]] = self.postProcessing(pr_depth, un_range_, pred, px_, py_)

            # vehicle ego motion - no need to preprocess
            vehicle_ego_mask = torch.zeros((40000), dtype=torch.bool)
            pts_within_range = un_range_ < 3.0
            vehicle_ego_mask[:pts[0]] = pts_within_range
            batched_labels_knn[ii, vehicle_ego_mask] = batched_labels_pred[ii, vehicle_ego_mask]

        return labels,batched_labels_pred,batched_labels_knn

