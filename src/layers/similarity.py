import torch
import torch.nn.functional as F

def normalize_feature(F_map):
    return F.normalize(F_map, p=2, dim=1)

def feature_distance(student_feat, teacher_feat):
    """
    Eq.2-3: Pixel-wise normalized feature difference
    """
    s_norm = normalize_feature(student_feat)
    t_norm = normalize_feature(teacher_feat)
    return 0.5 * ((s_norm - t_norm) ** 2).mean()
