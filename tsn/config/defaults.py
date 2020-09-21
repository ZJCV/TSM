from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.MODALITY = 'RGB'
_C.DATASETS.NUM_SEGS = 3
# train
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = 'HMDB51'
_C.DATASETS.TRAIN.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TRAIN.ANNOTATION_DIR = 'data/hmdb51'
# for hmdb51 and ucf101
_C.DATASETS.TRAIN.SPLIT = 1
# test
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = 'HMDB51'
_C.DATASETS.TEST.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TEST.ANNOTATION_DIR = 'data/hmdb51'
# for hmdb51 and ucf101
_C.DATASETS.TEST.SPLIT = 1

# ---------------------------------------------------------------------------- #
# Transform
# ---------------------------------------------------------------------------- #
_C.TRANSFORM = CN()
# HxWxC
_C.TRANSFORM.INPUT_SIZE = (112, 112, 3)
_C.TRANSFORM.MEAN = (0.485, 0.456, 0.406)  # (0.5, 0.5, 0.5)
_C.TRANSFORM.STD = (0.229, 0.224, 0.225)  # (0.5, 0.5, 0.5)

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.PRETRAINED = ""

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.BACKBONE.PARTIAL_BN = False
_C.MODEL.BACKBONE.TORCHVISION_PRETRAINED = True

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = 'TSNHead'
_C.MODEL.HEAD.FEATURE_DIMS = 2048
_C.MODEL.HEAD.DROPOUT = 0
_C.MODEL.HEAD.NUM_CLASSES = 51

_C.MODEL.RECOGNIZER = CN()
_C.MODEL.RECOGNIZER.NAME = 'TSNRecognizer'

_C.MODEL.CONSENSU = CN()
_C.MODEL.CONSENSU.NAME = 'AvgConsensus'

_C.MODEL.CRITERION = CN()
_C.MODEL.CRITERION.NAME = 'crossentropy'

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'sgd'
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 3e-5
# for sgd
_C.OPTIMIZER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# LR_Scheduler
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = 'multistep_lr'
# for SteLR
_C.LR_SCHEDULER.STEP_SIZE = 10000
# for MultiStepLR
_C.LR_SCHEDULER.MILESTONES = [15000, 25000]
_C.LR_SCHEDULER.GAMMA = 0.1
# for CosineAnnlearingLR
_C.LR_SCHEDULER.MINIMAL_LR = 3e-5
# for Warmup
_C.LR_SCHEDULER.WARMUP = False
_C.LR_SCHEDULER.ITERATION = 400
_C.LR_SCHEDULER.MULTIPLIER = 1.0

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.NAME = 'TSN.train'
_C.TRAIN.MAX_ITER = 30000

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.INFER = CN()
_C.INFER.NAME = 'TSN.infer'

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = 'outputs/'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
