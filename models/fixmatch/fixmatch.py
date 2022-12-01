import pickle

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os
import contextlib
import numpy as np

from .fixmatch_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss, EMA, Bn_Controller

from sklearn.metrics import *
from copy import deepcopy
import neptune.new as neptune

class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FixMatch, self).__init__()

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
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None
        self.best_eval_acc = None
        self.best_eval_iter = None
        self.run = None
        self.temperature = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.best_eval_acc = None
        self.best_eval_iter = None
        self.save_path = None

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

        # eval for once to verify if the checkpoint is loaded correctly
        # if args.resume == True:
        #     eval_dict = self.evaluate(args=args)
        #     print(eval_dict)

        if self.it < args.num_train_iter:
            self.run = neptune.init(
                project="supcon-team/cvpr2023",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMmNhNTkxOC0xYzQxLTRjNGQtOTE0YS00YWQ5ZGUxYjZiN2UifQ==",
                tags=args.tags + "_" + str(args.gpu)
            )  # your credentials

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
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

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                                                                       logits_x_ulb_w,
                                                                       'ce', T, p_cutoff,
                                                                       use_hard_labels=args.hard_label)

                total_loss = sup_loss + self.lambda_u * unsup_loss

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
            # tb_dict['train/sup_loss'] = sup_loss.detach()
            # tb_dict['train/unsup_loss'] = unsup_loss.detach()
            # tb_dict['train/total_loss'] = total_loss.detach()
            # tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            # tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            # tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            # tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

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
                self.run['train/total_loss'].log(total_loss)
                self.run['mask_ratio'].log(mask.mean())
                self.run['global_best_acc'].log(
                    self.best_eval_acc)  # 全局Top-1准确率：由于在超算平台上计算，会多次kill，多次读取checkpoints文件继续跑，所以维护一个全局Top-1准确率
                self.run['global_best_epoch'].log(self.best_eval_iter / self.num_eval_iter)  # 取得全局Top-1准确率所在的Epoch
                self.run['epochs'].log(self.it / self.num_eval_iter)  # 当前epoch
                self.run['eval/top-1-acc'].log(tb_dict['eval/top-1-acc'])  # 当前epoch下的最优准确率
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
            logits = self.model(x)
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

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.run['eval/loss'].log(total_loss / total_num)
        self.run['eval/top-1-acc'].log(top1)
        self.run['eval/top-5-acc'].log(top5)
        self.run['eval/precision'].log(precision)
        self.run['eval/recall'].log(recall)
        self.run['eval/F1'].log(F1)
        self.run['eval/AUC'].log(AUC)
        # np.set_printoptions(threshold=np.inf)
        # self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        # return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1}
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
