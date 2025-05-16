import os
# /kaggle/working/dataset
# /kaggle/working/model_checkpoint/best_model.keras
BASE_DATA_PATH = os.path.join('/', 'kaggle', 'working', 'dataset')
AUDIO_PATH: str = os.path.join(BASE_DATA_PATH, 'audio')
TRAIN_IMAGE_PATH: str = os.path.join(BASE_DATA_PATH, 'new_data', 'train')
TEST_IMAGE_PATH: str = os.path.join(BASE_DATA_PATH, 'new_data', 'test')

MODEL_CHECKPOINT_PATH: str = os.path.join('/', 'working', 'model_checkpoint')
