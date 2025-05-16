import os

# /home/bintang/Desktop/Skripsi/data
BASE_PATH = os.path.join(r'/', 'home', 'bintang', 'Desktop', 'Skripsi')
BASE_DATA_PATH = os.path.join(BASE_PATH, 'data')
AUDIO_PATH: str = os.path.join(BASE_DATA_PATH, 'audio')
TRAIN_IMAGE_PATH: str = os.path.join(BASE_DATA_PATH, 'new_data', 'train')
TEST_IMAGE_PATH: str = os.path.join(BASE_DATA_PATH, 'new_data', 'test')

MODEL_CHECKPOINT_PATH: str = os.path.join(BASE_PATH, 'model_checkpoint')

