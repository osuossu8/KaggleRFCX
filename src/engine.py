import numpy as np
import sys
sys.path.append("/root/workspace/KaggleRFCX")

import torch
from sklearn import metrics

from tqdm import tqdm
tqdm.pandas()


def _one_sample_positive_class_precisions(scores: np.ndarray, truth: np.ndarray):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def lwlrap(truth: np.ndarray, scores: np.ndarray):
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(
            scores[sample_num, :], truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    if float(np.sum(labels_per_class)) > 0:
        weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    else:
        weight_per_class = np.zeros_like(labels_per_class)

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class


def _lwlrap_sklearn(truth, scores):
    """Reference implementation from https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8"""
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def f1_loss(y_true, y_pred, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    
    y_pred = y_pred > 0.5
    
    tp = (y_true * y_pred).sum()#.to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred))#.sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum()#.to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum()#.to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    #f1.requires_grad = is_training #pytorchで利用する際は自動微分が可能 requires_gradで
    return f1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        score_class, weight = lwlrap(np.array(self.y_true), np.array(self.y_pred))
        self.score = (score_class * weight).sum()
        # self.score = _lwlrap_sklearn(np.array(self.y_true), np.array(self.y_pred)) #(score_class * weight).sum()
        self.f1_score = f1_loss(np.array(self.y_true), np.array(self.y_pred))
        return {
            "lwlrap" : self.score,
            "f1_score": self.f1_score
        }


def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch):
    losses = AverageMeter()
    scores = MetricMeter()

    model.train()
    t = tqdm(loader)
    for i, sample in enumerate(t):
        optimizer.zero_grad()
        input = sample['image'].to(args.device)
        target = sample['target'].to(args.device)
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if scheduler and args.step_scheduler:
            scheduler.step()

        bs = input.size(0)
        scores.update(target, torch.max(output['framewise_output'], dim=1)[0])
        # scores.update(target, torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]))
        losses.update(loss.item(), bs)

        t.set_description(f"Train E:{epoch} - Loss{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg

        
def valid_epoch(args, model, loader, criterion, epoch):
    losses = AverageMeter()
    scores = MetricMeter()
    model.eval()
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample['image'].to(args.device)
            target = sample['target'].to(args.device)
            output = model(input)
            loss = criterion(output, target)

            bs = input.size(0)
            scores.update(target, torch.max(output['framewise_output'], dim=1)[0])
            # scores.update(target, torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]))
            losses.update(loss.item(), bs)
            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg


def valid_epoch_v2(args, model, loader, criterion, epoch):
    losses = AverageMeter()
    scores = MetricMeter()
    model.eval()
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            o = {}
            input = sample['image'].to(args.device)
            bs, seq, w = input.shape
            input = input.reshape(bs*seq, w)
            target = sample['target'].to(args.device)
            output = model(input)

            tmp = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
            tmp = tmp.reshape(bs, seq, -1)
            tmp, _ = torch.max(tmp, dim=1)
            o['clipwise_output'] = tmp
            
            loss = criterion(o, target)    
            # loss = criterion(output, target)

            bs = input.size(0)
            scores.update(target, tmp)
            losses.update(loss.item(), bs)
            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg


def test_epoch(args, model, loader):
    model.eval()
    pred_list = []
    id_list = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample["image"].to(args.device)
            bs, seq, w = input.shape
            input = input.reshape(bs*seq, w)
            id = sample["id"]
            output = model(input)
            # output = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
            output = torch.max(output['framewise_output'], dim=1)[0]
            output = output.reshape(bs, seq, -1)
            output = torch.sum(output, dim=1)
            #output, _ = torch.max(output, dim=1)
            output = output.cpu().detach().numpy().tolist()
            pred_list.extend(output)
            id_list.extend(id)
    
    return pred_list, id_list
