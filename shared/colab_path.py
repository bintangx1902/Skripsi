import os
# /content/data/dataset
BASE_DATA_PATH = os.path.join('/', 'content', 'data')
AUDIO_PATH: str = os.path.join(BASE_DATA_PATH, 'audio')
TRAIN_IMAGE_PATH: str = os.path.join(BASE_DATA_PATH, 'new_data', 'train')
TEST_IMAGE_PATH: str = os.path.join(BASE_DATA_PATH, 'new_data', 'test')

MODEL_CHECKPOINT_PATH: str = os.path.join('/', 'content', 'model_checkpoint')

