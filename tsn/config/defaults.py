from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.NAME = 'TSN.train'
_C.TRAIN.MAX_ITER = 100000
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_STEP = 2500
_C.TRAIN.EVAL_STEP = 2500

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.INFER = CN()
_C.INFER.NAME = 'TSN.infer'

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.BACKBONE = 'resnet50'
_C.MODEL.CONSENSUS = 'avg'
_C.MODEL.PARTIAL_BN = False
_C.MODEL.PRETRAINED = False
# HxWxC
_C.MODEL.INPUT_SIZE = (112, 112, 3)
_C.MODEL.NUM_CLASSES = 51

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.CRITERION = CN()
_C.CRITERION.NAME = 'crossentropy'

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
_C.LR_SCHEDULER.STEP_SIZE = 40000
# for MultiStepLR
_C.LR_SCHEDULER.MILESTONES = [25000, 60000]
_C.LR_SCHEDULER.GAMMA = 0.1

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.MODALITY = ('RGB', 'RGBDiff')

_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = 'HMDB51'
# for hmdb51 and ucf101
_C.DATASETS.TRAIN.SPLITS = (1, 2, 3)
_C.DATASETS.TRAIN.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TRAIN.ANNOTATION_DIR = 'data/hmdb51'

_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = 'HMDB51'
# for hmdb51 and ucf101
_C.DATASETS.TEST.SPLITS = (1, 2, 3)
_C.DATASETS.TEST.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TEST.ANNOTATION_DIR = 'data/hmdb51'

# ---------------------------------------------------------------------------- #
# Transform
# ---------------------------------------------------------------------------- #
_C.TRANSFORM = CN()
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
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = 'outputs/resnet50_hmdb51'
