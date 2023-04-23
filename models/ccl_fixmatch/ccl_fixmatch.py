import torch

import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
import contextlib
import numpy as np

from .ccl_fixmatch_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss, EMA, Bn_Controller

from sklearn.metrics import *
from copy import deepcopy


class CCL_FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, hard_label=True, num_eval_iter=1024, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            hard_label: If True, consistency regularization use a hard pseudo label.
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(CCL_FixMatch, self).__init__()

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
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None
        self.best_eval_acc = None
        self.best_eval_iter = None

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
        # Multiple scheduling mechanism for timeout termination adapted to supercomputing platform
        if self.best_eval_acc is None:
            self.best_eval_acc = 0.0
        if self.best_eval_iter is None:
            self.best_eval_iter = 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            device = torch.device('cuda', index=args.gpu)
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(
                args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, features = self.model(inputs)
                logits_x = logits[:num_lb]
                logits_u_w, logits_u_s = logits[num_lb:].chunk(2)
                features_x = features[:num_lb]
                features_u_w, features_u_s = features[num_lb:].chunk(2)
                del logits
                del features
                sup_loss = ce_loss(logits_x, y_lb, reduction='mean')

                pseudo_label, unsup_loss, over_thre_mask, less_thre_mask, p_targets_u = consistency_loss(
                    logits_u_s, logits_u_w, 'ce', args.T, args.p_cutoff, use_hard_labels=args.hard_label)

                over_thre_features = torch.cat(
                    (features_x, features_u_w[over_thre_mask]), 0)
                over_thre_features = torch.cat(
                    (over_thre_features, features_u_s[over_thre_mask]), 0)
                top1_high = torch.cat((y_lb, p_targets_u[over_thre_mask]), 0)
                top1_high = torch.cat(
                    (top1_high, p_targets_u[over_thre_mask]), 0).view(-1, 1)
                less_thre_features = torch.cat(
                    (features_u_w[less_thre_mask], features_u_s[less_thre_mask]), 0)
                top1_low = torch.cat(
                    (p_targets_u[less_thre_mask], p_targets_u[less_thre_mask]), 0).view(-1, 1)

                less_thre_features = torch.cat(
                    (features_u_w[less_thre_mask], features_u_s[less_thre_mask]), 0)
                top1_low = torch.cat(
                    (p_targets_u[less_thre_mask], p_targets_u[less_thre_mask]), 0).view(-1, 1)

                # get complementary labels
                _, min_k_pseudo_label = pseudo_label.topk(
                    dim=1, k=int(args.k*args.num_classes), largest=False)
                min_k_low = torch.cat(
                    (min_k_pseudo_label[less_thre_mask], min_k_pseudo_label[less_thre_mask]), 0)

                # constrcut positive pairs
                pos1 = torch.eq(top1_high, top1_high.T).float().to(device)
                pos1.scatter_(dim=1, index=torch.arange(
                    pos1.shape[0]).view(-1, 1).to(device), value=0)
                pos2 = torch.zeros(
                    top1_high.shape[0], top1_low.shape[0]).to(device)
                pos3 = pos2.T
                pos4 = torch.zeros(
                    top1_low.shape[0], top1_low.shape[0]).to(device)
                ids = torch.tensor(
                    [i for i in range(int(top1_low.shape[0] / 2))])
                view_from_same_image_index = torch.cat(
                    (ids + top1_low.shape[0] / 2, ids), 0).view(-1, 1).to(device)
                pos4.scatter_(
                    dim=1, index=view_from_same_image_index.long(), value=1)
                pos_mask = torch.cat(
                    (torch.cat((pos1, pos2), 1), torch.cat((pos3, pos4), 1)), 0)

                # construct logits_mask, which contains positive pairs and negative pairs
                if top1_low.shape[0] != 0 and top1_high.shape[0] != 0:
                    logits_mask1 = torch.ones(top1_high.shape[0], top1_high.shape[0]).scatter_(
                        dim=1, index=torch.arange(top1_high.shape[0]).view(-1, 1), value=0).to(device)

                    logits_mask2 = torch.sum(
                        torch.eq(top1_high.repeat(1, int(args.k*args.num_classes)).repeat(1, min_k_low.shape[0])
                                 .view(top1_high.shape[0], min_k_low.shape[0], -1),
                                 min_k_low.repeat(top1_high.shape[0], 1)
                                 .view(top1_high.shape[0], min_k_low.shape[0], -1)), dim=2)

                    logits_mask3 = logits_mask2.T

                    logits_mask = torch.cat(
                        (torch.cat((logits_mask1, logits_mask2), 1), torch.cat((logits_mask3, pos4), 1)), 0)
                elif top1_low.shape[0] == 0:
                    logits_mask = torch.ones(top1_high.shape[0], top1_high.shape[0]).scatter_(
                        dim=1, index=torch.arange(top1_high.shape[0]).view(-1, 1), value=0).to(device)
                elif top1_high.shape[0] == 0:
                    logits_mask = pos4

                # feature affinity
                total_features = torch.cat(
                    (over_thre_features, less_thre_features), 0)
                anchor_dot_contrast = torch.div(torch.matmul(
                    total_features, total_features.T), args.temperature)
                logits_max, _ = torch.max(
                    anchor_dot_contrast, dim=1, keepdim=True)
                logits_con = anchor_dot_contrast - logits_max.detach()

                exp_logits = torch.exp(logits_con) * logits_mask
                log_prob = logits_con - \
                    torch.log(exp_logits.sum(1, keepdim=True))

                mean_log_prob_pos = (pos_mask * log_prob).mean(1)
                con_loss = - mean_log_prob_pos
                con_loss = con_loss.view(pos_mask.shape[0])
                N_nonzeor = torch.nonzero(pos_mask.sum(1)).shape[0]
                con_loss = con_loss.sum() / N_nonzeor

                if torch.isnan(con_loss):
                    con_loss = torch.zeros(1).to(device)
                total_loss = sup_loss + unsup_loss + con_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/con_loss'] = con_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(
                end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 1024 steps and best model for each 1024 steps
            if self.it != 0 and self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] >= self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    self.best_eval_iter = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_eval_iter} iters")

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('checkpoint.pth.tar', save_path)
                    if self.it == self.best_eval_iter:
                        self.save_model('model_best.pth', save_path)
                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': self.best_eval_acc,
                         'eval/best_it': self.best_eval_iter})
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
        # np.set_printoptions(threshold=np.inf)
        # self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5, 'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

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
        xy = [[v[offsets[p]:offsets[p + 1]]
               for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
