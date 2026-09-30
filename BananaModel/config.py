import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'
CHECKPOINT_PATH = 'checkpoints/best_banana_model.pth'

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 4