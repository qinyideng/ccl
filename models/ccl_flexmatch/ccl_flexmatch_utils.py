import torch
import math
import torch.nn.functional as F
import numpy as np

from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def consistency_loss(logits_s, logits_w, class_acc, p_target, p_model, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff).float()
        over_thre_mask = max_probs.ge(
            p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx])))
        less_thre_mask = max_probs.lt(
            p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx])))

        if use_hard_labels:
            masked_loss = ce_loss(
                logits_s, max_idx, use_hard_labels, reduction='none') * over_thre_mask.float()
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label,
                                  use_hard_labels) * over_thre_mask.float()
        return pseudo_label, masked_loss.mean(), over_thre_mask, less_thre_mask, max_idx.long()

    else:
        assert Exception('Not Implemented consistency_loss')
