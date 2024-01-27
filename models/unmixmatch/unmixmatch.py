import contextlib
import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler

from .unmixmatch_utils import Get_Scalar, one_hot, mixup_one_target, Bn_Controller
from .criterions import SupConLoss, ce_loss


class UnMixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, num_eval_iter=1000, tb_log=None, logger=None):
        super(UnMixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.best_eval_acc = 0.0
        self.best_it = 0
        self.logger = logger
        self.contrastive_loss = SupConLoss

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        print(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        contrastive_criterion = self.contrastive_loss(temperature=args.contrastive_temperature)

        # for gpu profiling
        if torch.cuda.is_available():
            start_batch = torch.cuda.Event(enable_timing=True)
            end_batch = torch.cuda.Event(enable_timing=True)
            start_run = torch.cuda.Event(enable_timing=True)
            end_run = torch.cuda.Event(enable_timing=True)

            start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        if not os.path.exists(dist_file_name):
            p_target = torch.ones(args.num_classes) / args.num_classes
        else:
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
        if torch.cuda.is_available():
            p_target = p_target.cuda(args.gpu)
        print('p_target:', p_target)

        p_model = None

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (_, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(self.loader_dict['train_lb'],
                                                                                          self.loader_dict[
                                                                                              'train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            num_rot = x_ulb_s1_rot.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            if torch.cuda.is_available():
                x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(args.gpu), x_ulb_s2.cuda(args.gpu)
                x_ulb_s1_rot = x_ulb_s1_rot.cuda(args.gpu)
                rot_v = rot_v.cuda(args.gpu)
                y_lb = y_lb.cuda(args.gpu)

            # inference and calculate sup/unsup losses
            with amp_cm():
                bsz = x_ulb_s1.shape[0]
                if args.contrast_factor == 'strong':
                    input_images = torch.cat([x_ulb_s1, x_ulb_s2], dim=0)
                    _, _, features = self.model(input_images, return_projection=True)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contrastive_loss = contrastive_criterion(features)
                else:
                    input_images = torch.cat([x_ulb_w, x_ulb_s1, x_ulb_s2], dim=0)
                    _, _, features = self.model(input_images, return_projection=True)
                    f1, f2, f3 = torch.split(features, [bsz, bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                    contrastive_loss = contrastive_criterion(features)

                with torch.no_grad():
                    self.bn_controller.freeze_bn(self.model)
                    logits_x_ulb_w = self.model(x_ulb_w)[0]

                    self.bn_controller.unfreeze_bn(self.model)
                    # hyper-params for update
                    T = self.t_fn(self.it)

                    prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

                    if p_model is None:
                        p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                    else:
                        p_model = p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001

                    # Distribution alignment
                    prob_x_ulb = prob_x_ulb * p_target / p_model
                    prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))

                    sharpen_prob_x_ulb = prob_x_ulb ** (1 / T)
                    sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                    # mix up
                    mixed_inputs = torch.cat((x_lb, x_ulb_s1, x_ulb_s2, x_ulb_w))
                    input_labels = torch.cat(
                        [one_hot(y_lb, args.num_classes, args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb,
                         sharpen_prob_x_ulb], dim=0)

                    mixed_x, mixed_y, _ = mixup_one_target(mixed_inputs, input_labels,
                                                           args.gpu,
                                                           args.alpha,
                                                           is_bias=True)

                    # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                    mixed_x = list(torch.split(mixed_x, num_lb))
                    mixed_x = self.interleave(mixed_x, num_lb)

                # calculate BN only for the first batch
                logits = [self.model(mixed_x[0])[0]]

                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt)[0])

                u1_logits = self.model(x_ulb_s1)[0]
                logits_rot = self.model(x_ulb_s1_rot)[1]
                logits = self.interleave(logits, num_lb)
                self.bn_controller.unfreeze_bn(self.model)

                logits_x = logits[0]
                logits_u = torch.cat(logits[1:])

                # calculate rot loss with w_rot
                rot_loss = ce_loss(logits_rot, rot_v, reduction='mean')
                rot_loss = rot_loss.mean()

                # sup loss
                sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
                sup_loss = sup_loss.mean()

                total_loss = sup_loss + args.w_rot * rot_loss + args.w_contrastive * contrastive_loss

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
            self.model.zero_grad()

            if torch.cuda.is_available():
                end_run.record()
                torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {'train/sup_loss': sup_loss.detach(), 'train/unsup_loss': contrastive_loss.detach(), 'train/total_loss': total_loss.detach(), 'lr': self.optimizer.param_groups[0]['lr']}

            if self.it % 100 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                self.save_model('latest_model.pth', save_path)
            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                if tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    self.best_it = self.it

                print(f"{self.it} iteration, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_it} iters")

            self.it += 1
            del tb_dict
            if torch.cuda.is_available():
                start_batch.record()

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        try:
            os.makedirs(args.save_dir + '/eval_acc', exist_ok=True)
            with open(os.path.join(args.save_dir + '/eval_acc', args.save_name[:-2] + '.txt'), 'a') as f:
                f.write(args.save_name + ' ' + str(round(self.best_eval_acc * 100, 2)) + '\n')
        except:
            pass
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            if torch.cuda.is_available():
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
        cf_mat = confusion_matrix(y_true, y_pred)
        print('confusion matrix:\n' + np.array_str(cf_mat))
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'best_eval_acc': self.best_eval_acc,
                    'best_it': self.best_it,},
                    save_filename)

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        print('Loading save model from: ' + str(load_path))
        self.model.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        try:
            self.best_eval_acc = checkpoint['best_eval_acc']
            self.best_it = checkpoint['best_it']
        except:
            print('no best eval acc found')

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
