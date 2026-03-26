import torch.nn as nn
from ..backbone.student_encoder import StudentEncoder


class Student1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StudentEncoder()

    def forward(self, x):
        return self.encoder(x)
