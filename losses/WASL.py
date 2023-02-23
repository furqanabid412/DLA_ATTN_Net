import torch
from torch import nn


class WAsymmetricLoss(nn.Module):
    def __init__(self, class_weights ,gamma_neg=2, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(WAsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.classes = len(class_weights)
        self.class_weights = class_weights[None, :, None, None]


    def one_hot_enc(self, y, device='cpu'):
        b, h, w = y.shape
        y = y.view(b * h * w)
        target = torch.zeros(self.classes, len(y))
        for class_id in range(self.classes):
            target[class_id, :] = y == class_id
        target = target.view(self.classes,b,h,w)
        target = torch.swapaxes(target,0,1)

        return target.to(device)

    def forward(self, x, y):

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)


        device = 'cpu' if y.get_device() < 0 else y.get_device()
        y = self.one_hot_enc(y, device=device)
        self.class_weights = self.class_weights.to(device)

        # adaptive weights calculations
        wt = torch.multiply(self.class_weights, y)
        mask = torch.logical_not(y).to(torch.uint8)
        adap_wt = wt + mask

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # loss = self.XE(x,y)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # multiply with weights
        adap_loss = loss * adap_wt
        # b, cl, h, w = loss.shape
        # return -loss.sum() /( b * cl * h * w)
        return -adap_loss.sum()