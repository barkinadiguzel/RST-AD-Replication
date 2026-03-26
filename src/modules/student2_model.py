import torch.nn as nn
from ..backbone.student_encoder import StudentEncoder
from ..layers.attention import AttentionBlock
from ..layers.decoder_block import DecoderBlock


class Student2Model(nn.Module):
    def __init__(self, feature_dims=(256, 128, 64)):
        super().__init__()
        self.encoder = StudentEncoder()

        self.attn_blocks = nn.ModuleList([
            AttentionBlock(feature_dims[0]),
            AttentionBlock(feature_dims[1]),
            AttentionBlock(feature_dims[2]),
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(feature_dims[0], feature_dims[1], upsample=True),
            DecoderBlock(feature_dims[1], feature_dims[2], upsample=True),
            DecoderBlock(feature_dims[2], feature_dims[2], upsample=True),
        ])

    def forward(self, x, teacher_feats):
        student_feats = self.encoder(x)
        out = student_feats[-1]

        recon_feats = []
        teacher_rev = teacher_feats[::-1]  

        for i in range(3):
            out = self.attn_blocks[i](out, teacher_rev[i])
            out = self.decoder_blocks[i](out)
            recon_feats.append(out)

        return recon_feats
