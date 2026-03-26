import torch
import torch.nn as nn
from ..backbone.teacher_resnet import TeacherResNet18, TeacherResNet50
from ..modules.student1_model import Student1Model  
from ..modules.student2_model import Student2Model  
from ..modules.discriminative_net import DiscriminativeNet

class RSTADModel(nn.Module):
    def __init__(self, feature_dims_student1=[64, 128, 256],
                       feature_dims_student2=[64, 128, 256]):
        super().__init__()
        # Teachers
        self.teacher1 = TeacherResNet18()
        self.teacher2 = TeacherResNet50()
        # Students
        self.student1 = Student1Model(feature_dims_student1)
        self.student2 = Student2Model(feature_dims_student2)
        # Discriminative network
        self.discriminator = DiscriminativeNet(in_channels=2)  

    def forward(self, x):
        t1_feats = self.teacher1(x)
        t2_feats = self.teacher2(x)

        s1_out = self.student1(x, t1_feats)  
        s2_out = self.student2(x, t2_feats) 

        anomaly_map1 = torch.mean(torch.abs(s1_out - t1_feats[-1]), dim=1, keepdim=True)
        anomaly_map2 = torch.mean(torch.abs(s2_out - t2_feats[-1]), dim=1, keepdim=True)

        combined_maps = torch.cat([anomaly_map1, anomaly_map2], dim=1)
        refined_map = self.discriminator(combined_maps)

        return {
            "s1_out": s1_out,
            "s2_out": s2_out,
            "anomaly_map1": anomaly_map1,
            "anomaly_map2": anomaly_map2,
            "refined_map": refined_map
        }
