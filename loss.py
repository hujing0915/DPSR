import torch
import torch.nn as nn
import torch.nn.functional as F

########################### FREE model ###########################
def Other_label(labels, num_classes):
    index = torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    other_labels = labels+index
    other_labels[other_labels >= num_classes] = other_labels[other_labels >= num_classes]-num_classes
    return other_labels

class TripCenterLoss_margin(nn.Module):
    def __init__(self, num_classes=10, feat_dim=312, use_gpu=True):
        super(TripCenterLoss_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, margin, incenter_weight):
        other_labels = Other_label(labels, self.num_classes)
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(
            mat1=x,
            mat2=self.centers.t(),
            beta=1,
            alpha=-2
        )
        classes = torch.arange(self.num_classes).long()

        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]

        other_labels = other_labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_other = other_labels.eq(classes.expand(batch_size, self.num_classes))
        dist_other = distmat[mask_other]

        dynamic_margin = self.adjust_margin(margin, dist, dist_other)
        loss = torch.max(dynamic_margin + incenter_weight * dist - (1 - incenter_weight) * dist_other,
                         torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

    def adjust_margin(self, margin, dist, dist_other):
        avg_dist_in = dist.mean()
        avg_dist_out = dist_other.mean()
        if avg_dist_out > avg_dist_in:
            adjusted_margin = margin * (1 + (avg_dist_out - avg_dist_in).item())
        else:
            adjusted_margin = margin

        return adjusted_margin

class TripCenterLoss_min_margin(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(TripCenterLoss_min_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, margin, incenter_weight):
        batch_size = x.size(0)

        distmat = torch.cdist(x, self.centers, p=1)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]

        other = torch.FloatTensor(batch_size, self.num_classes - 1).cuda()
        for i in range(batch_size):
            other[i] = distmat[i, mask[i, :] == 0]
        dist_min, _ = other.min(dim=1)

        dynamic_margin = self.adjust_margin(margin, dist, dist_min)
        loss = torch.max(dynamic_margin + dist - dist_min,
                         torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

    def adjust_margin(self, margin, dist, dist_other):
        avg_dist_in = dist.mean()
        avg_dist_out = dist_other.mean()
        if avg_dist_out > avg_dist_in:
            adjusted_margin = margin * (1 + (avg_dist_out - avg_dist_in).item())
        else:
            adjusted_margin = margin
        return adjusted_margin

########################### Ours model ###########################
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss
