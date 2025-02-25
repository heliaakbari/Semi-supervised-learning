# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):
    """
        FixMatch algorithm with Triplet Loss (https://arxiv.org/abs/2001.07685).
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # FixMatch-specific arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, lambda_t=args.lambda_t)
        # Triplet loss function
        self.triplet_loss = nn.TripletMarginLoss(margin=args.triplet_margin, p=2)

    def init(self, T, p_cutoff, hard_label=True, lambda_t=0.1):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.lambda_t = lambda_t  # Weight for triplet loss

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # Inference and calculate losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                print("Logits shape:", outputs.shape)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']

        feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

        # Compute supervised classification loss
        sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        # Compute probabilities for unlabeled samples
        probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

        # Apply distribution alignment if enabled
        if self.registered_hook("DistAlignHook"):
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

        # Compute mask
        mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

        # Generate pseudo-labels
        pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                      logits=probs_x_ulb_w,
                                      use_hard_label=self.use_hard_label,
                                      T=self.T,
                                      softmax=False)

        # Compute unsupervised consistency loss
        unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                           pseudo_label,
                                           'ce',
                                           mask=mask)

        # Combine total loss
        triplet_loss = self.triplet_loss(feats_x_lb, feats_x_ulb_w, feats_x_ulb_s)

        total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_t * triplet_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         triplet_loss=triplet_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())

        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--lambda_t', float, 0.1),  # Weight for triplet loss
            SSL_Argument('--triplet_margin', float, 1.0),  # Triplet loss margin
        ]
