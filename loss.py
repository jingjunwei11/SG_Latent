# coding: utf-8
"""
Module to implement training loss
"""

from torch import nn, Tensor


class RegLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
            self.criterion_L1 = nn.L1Loss()

        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask
        # loss = self.criterion(preds_masked, targets_masked) + 0.1 * self.criterion_L1(preds_masked, targets_masked)
        loss = self.criterion(preds_masked, targets_masked)
        # Calculate loss just over the masked predictions
        # body_loss = self.criterion(preds_masked[:, :, :8 * 3], targets_masked[:, :, :8 * 3])
        # wrist_loss = self.criterion(preds_masked[:, :, 8 * 3:9 * 3],
        #                             targets_masked[:, :, 8 * 3:9 * 3]) + self.criterion(
        #     preds_masked[:, :, 29 * 3:30 * 3], targets_masked[:, :, 29 * 3:30 * 3])
        # left_hand_loss = self.criterion(preds_masked[:, :, 8 * 3:29 * 3], targets_masked[:, :, 8 * 3:29 * 3])
        # right_hand_loss = self.criterion(preds_masked[:, :, 29 * 3:], targets_masked[:, :, 29 * 3:])
        # w_b = 1
        # w_w = 1
        # w_l = 1
        # w_r = 1
        # loss = w_b * body_loss + w_w * wrist_loss + w_l * left_hand_loss + w_r * right_hand_loss

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # standard xent loss
        self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                    reduction='sum')

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        # targets: indices with batch*seq_len
        targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)

        return loss


class HuberLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(HuberLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.HuberLoss()
        elif self.loss == "mse":
            self.criterion = nn.HuberLoss()
            self.criterion_L1 = nn.HuberLoss()

        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.HuberLoss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask
        loss = self.criterion(preds_masked, targets_masked)

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss