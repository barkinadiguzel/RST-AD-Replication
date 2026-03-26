def extract_multi_res_features(teacher, student, x):
    t_feats = teacher(x)
    s_feats = student.encoder(x)
    return t_feats, s_feats
