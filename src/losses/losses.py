import torch
import torch.nn.functional as F


def distillation_loss(student_feats, teacher_feats, eps: float = 1e-6):
    loss = 0.0
    for s, t in zip(student_feats, teacher_feats):
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)

        s = F.normalize(s, p=2, dim=1, eps=eps)
        t = F.normalize(t, p=2, dim=1, eps=eps)
        loss = loss + 0.5 * (s - t).pow(2).mean()
    return loss


def reconstruction_loss(recon_feats, teacher_feats, eps: float = 1e-6):
    loss = 0.0
    teacher_rev = teacher_feats[::-1]
    for r, t in zip(recon_feats, teacher_rev):
        if r.shape[-2:] != t.shape[-2:]:
            r = F.interpolate(r, size=t.shape[-2:], mode="bilinear", align_corners=False)

        r = F.normalize(r, p=2, dim=1, eps=eps)
        t = F.normalize(t, p=2, dim=1, eps=eps)
        loss = loss + 0.5 * (r - t).pow(2).mean()
    return loss


def focal_loss(pred, target, gamma: float = 2.0, eps: float = 1e-6)
    pred = pred.clamp(eps, 1.0 - eps)
    loss = -target * ((1.0 - pred) ** gamma) * torch.log(pred)
    return loss.mean()


def segmentation_loss(pred, target, gamma: float = 2.0, l1_weight: float = 1.0):
    return focal_loss(pred, target, gamma=gamma) + l1_weight * F.l1_loss(pred, target)


def total_loss(student1_feats,
               teacher1_feats,
               recon_feats,
               teacher2_feats,
               disc_pred=None,
               pseudo_mask=None,
               lambda_distill=1.0,
               lambda_recon=1.0,
               lambda_seg=1.0):
    loss = 0.0
    loss = loss + lambda_distill * distillation_loss(student1_feats, teacher1_feats)
    loss = loss + lambda_recon * reconstruction_loss(recon_feats, teacher2_feats)

    if disc_pred is not None and pseudo_mask is not None:
        loss = loss + lambda_seg * segmentation_loss(disc_pred, pseudo_mask)

    return loss
