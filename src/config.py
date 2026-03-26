import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_SIZE = 256

    TEACHER1 = "resnet18"
    TEACHER2 = "resnet50"
    STUDENT = "resnet18"

    FEATURE_LAYERS = ["layer1", "layer2", "layer3"]
  
    FEATURE_DIMS = [64, 128, 256]

    UPSAMPLE_MODE = "bilinear"
    ALIGN_CORNERS = False

    # loss weights 
    LAMBDA_KD = 1.0
    LAMBDA_REC = 1.0
    LAMBDA_SEG = 1.0

    FOCAL_GAMMA = 2

    USE_PSEUDO_ANOMALY = True

    USE_ATTENTION = True

    THRESHOLD = 0.5
