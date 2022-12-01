import torch

import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
import contextlib

from train_utils import ce_loss, EMA, Bn_Controller

from sklearn.metrics import *
from copy import deepcopy
import neptune.new as neptune


class CCSSL_FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, lambda_u, lambda_con, hard_label=True, num_eval_iter=1024, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(CCSSL_FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.lambda_u = lambda_u
        self.lambda_con = lambda_con
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None
        self.best_eval_acc = None
        self.best_eval_iter = None
        self.run = None

        self.it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):
        # ngpus_per_node = torch.cuda.device_count()
        ngpus_per_node = 2
        save_path = os.path.join(args.save_dir, args.save_name)
        if self.best_eval_acc is None:
            self.best_eval_acc = 0.0
        if self.best_eval_iter is None:
            self.best_eval_iter = 0
        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        if self.it < args.num_train_iter:
            self.run = neptune.init(
                project="supcon-team/cvpr2023",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMmNhNTkxOC0xYzQxLTRjNGQtOTE0YS00YWQ5ZGUxYjZiN2UifQ==",
                tags=args.tags + "_" + str(args.gpu)
            )  # your credentials

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s, x_ulb_s1) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s, x_ulb_s1 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu), x_ulb_s1.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s, x_ulb_s1))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, features = self.model(inputs)
                logits_x = logits[:num_lb]
                logits_u_w, logits_u_s, _ = logits[num_lb:].chunk(3)
                _, f_u_s1, f_u_s2 = features[num_lb:].chunk(3)
                del logits
                del features
                del _
                sup_loss = ce_loss(logits_x, y_lb, reduction='mean')

                pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                max_probs, p_targets_u = torch.max(pseudo_label, dim=-1)
                over_threshold_mask = max_probs.ge(args.p_cutoff).float()

                unsup_loss = (ce_loss(logits_u_s, p_targets_u, self.use_hard_label, reduction='none') * over_threshold_mask).mean()

                labels = p_targets_u
                features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)
                device = torch.device('cuda', index=args.gpu)
                if labels.shape[0] != 0:
                    if len(features.shape) < 3:
                        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                                         'at least 3 dimensions are required')
                    if len(features.shape) > 3:
                        features = features.view(features.shape[0], features.shape[1], -1)

                    batch_size = features.shape[0]

                    labels = labels.contiguous().view(-1, 1)
                    if labels.shape[0] != batch_size:
                        raise ValueError('Num of labels does not match num of features')
                    mask = torch.eq(labels, labels.T).float().to(device)
                    max_probs = max_probs.contiguous().view(-1, 1)
                    score_mask = torch.matmul(max_probs, max_probs.T)
                    mask = mask.mul(score_mask)

                    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
                    contrast_count = features.shape[1]
                    anchor_feature = contrast_feature
                    anchor_count = contrast_count

                    # compute logits
                    anchor_dot_contrast = torch.div(
                        torch.matmul(anchor_feature, contrast_feature.T),
                        args.temperature)
                    # for numerical stability
                    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                    logits = anchor_dot_contrast - logits_max.detach()

                    # tile mask
                    mask = mask.repeat(anchor_count, contrast_count)
                    # mask-out self-contrast cases
                    logits_mask = torch.scatter(
                        torch.ones_like(mask),
                        1,
                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                        0
                    )
                    mask = mask * logits_mask
                    # compute log_prob
                    exp_logits = torch.exp(logits) * logits_mask
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

                    # compute mean of log-likelihood over positive
                    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

                    # loss
                    con_loss = - mean_log_prob_pos
                    con_loss = con_loss.view(anchor_count, batch_size)
                    con_loss = con_loss.mean()
                else:
                    con_loss = sum(features.view(-1, 1)) * 0

                total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_con * con_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}

            if self.it != 0 and self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] >= self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    self.best_eval_iter = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_eval_iter} iters")

                self.run['train/sup_loss'].log(sup_loss)
                self.run['train/unsup_loss'].log(unsup_loss)
                self.run['train/con_loss'].log(con_loss)
                self.run['train/total_loss'].log(total_loss)
                self.run['mask_ratio'].log(mask.mean())
                self.run['global_best_acc'].log(self.best_eval_acc)
                self.run['global_best_epoch'].log(self.best_eval_iter / self.num_eval_iter)
                self.run['epochs'].log(self.it / self.num_eval_iter)
                self.run['eval/top-1-acc'].log(tb_dict['eval/top-1-acc'])
                self.run['eval/loss'].log(tb_dict['eval/loss'])

                if not args.multiprocessing_distributed or (
                        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('checkpoint.pth.tar', save_path)
                    if self.it == self.best_eval_iter:
                        self.save_model('model_best.pth', save_path)
                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_eval_iter})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        # cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.run['eval/loss'].log(total_loss / total_num)
        self.run['eval/top-1-acc'].log(top1)
        self.run['eval/top-5-acc'].log(top5)
        self.run['eval/precision'].log(precision)
        self.run['eval/recall'].log(recall)
        self.run['eval/F1'].log(F1)
        self.run['eval/AUC'].log(AUC)
        # self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model,
                    'best_eval_acc': self.best_eval_acc,
                    'best_eval_iter': self.best_eval_iter},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        self.best_eval_iter = checkpoint['best_eval_iter']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
