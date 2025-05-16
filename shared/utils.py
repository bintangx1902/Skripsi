import librosa
import numpy as np
import cv2
import tensorflow as tf
from collections import Counter


BATCH_SIZE = 64


def load_audio(file_content, target_sr):
    audio, sample_rate = librosa.load(file_content, sr=target_sr, mono=True)
    trim, _ = librosa.effects.trim(audio, top_db=20)
    wav_resampled = librosa.resample(trim, orig_sr=sample_rate, target_sr=target_sr)
    return wav_resampled, target_sr


def preprocess_audio(file_content, target_sr=32000, n_mels=128, fmin=80.0, fmax=7600.0, fixed_frames=110):
    """
    :param file_content: file path (full path or can be absolute path)  
    :param target_sr: if you have your own desired sampling rate
    :param n_mels: max length of mel-spectrogram
    :param fmin: minimum frequency
    :param fmax: maximum frequency
    :return: the mel-spectrogram cast as image with 3 channels
    """
    audio, sample_rate = load_audio(file_content, target_sr)
    # target_samples = int(np.ceil(len(audio) / sample_rate) * target_sr)
    # if len(audio) < target_samples:
    #     audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.min)

    if mel_spectrogram_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :fixed_frames]
    
    spectogram = np.expand_dims(mel_spectrogram_db, axis=-1)
    return np.repeat(spectogram, 3, axis=-1)


def preprocess_image(img_path, dsize: tuple = (128, 128), normalize=False, augment=False):
    """
    :param img_path: path to image (full path or can be absolute path)  
    :param dsize: new data size, default is 100 x 100 px
    :return: upsampled image to 100x100 pixels
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

    if augment:
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        factor = 1.0 + 0.1 * (np.random.rand() - 0.5)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)

    image = image.astype(np.float32) / 255.0

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
    return image


def plot_model(model: tf.keras.Model, show_shapes=False, show_layer_names=False):
    """
    :param model: the model you want to plot
    :param show_shapes: if you want to show the shape each layer of the model
    :param show_layer_names: if you want to show the layer names each layer of the model
    :return: 
    """
    tf.keras.utils.plot_model(model, show_shapes=show_shapes, show_layer_names=show_layer_names)


def stratified_downsample(x_img, x_audio, y_img, y_audio):
    """
    Downsamples the image dataset to match the total number of samples in the audio dataset
    while preserving all classes in both datasets.

    Args:
        x_img (np.array): Image inputs.
        x_audio (np.array): Audio inputs.
        y_img (np.array): Image labels.
        y_audio (np.array): Audio labels.

    Returns:
        Tuple of downsampled (x_img, x_audio, y_img, y_audio).
    """
    # Ensure seed for reproducibility
    np.random.seed(42)

    # Count the number of samples in each class for both datasets
    class_counts_img = Counter(y_img)
    class_counts_audio = Counter(y_audio)

    # Verify that all audio classes (8) are present
    audio_classes = set(y_audio)
    assert len(audio_classes) == 8, "Audio dataset must have 8 classes"

    # Verify that all image classes (7) are present
    img_classes = set(y_img)
    assert len(img_classes) == 7, "Image dataset must have 7 classes"

    # Prepare lists to store downsampled indices
    selected_indices_img = []

    # Downsample image data to match audio dataset total samples
    for cls in img_classes:
        # Find indices for this class in image data
        indices_img = np.where(y_img == cls)[0]

        # If this is the only way to sample while keeping all classes
        np.random.shuffle(indices_img)
        selected_indices_img.extend(indices_img)

    # Convert to numpy array for indexing
    selected_indices_img = np.array(selected_indices_img)

    # Downsample image data
    x_img_ds = x_img[selected_indices_img]
    y_img_ds = y_img[selected_indices_img]

    return x_img_ds, x_audio, y_img_ds, y_audio


def global_downsample_preserve_classes(x_img, x_audio, y_img, y_audio):
    """
    Randomly downsample the larger dataset to match the smaller one,
    ensuring that all original classes remain present.

    Returns:
        x_img_ds, x_audio_ds, y_img_ds, y_audio_ds
    """
    # Tentukan dataset yang lebih kecil
    n_img = len(y_img)
    n_audio = len(y_audio)
    target_samples = min(n_img, n_audio)

    def downsample(x, y, target_size):
        # Ulangi hingga mendapatkan subset yang masih mencakup semua kelas
        classes = set(y)
        while True:
            indices = np.random.choice(len(y), size=target_size, replace=False)
            y_subset = y[indices]
            if set(y_subset) == classes:
                return x[indices], y[indices]

    if n_img > n_audio:
        x_img_ds, y_img_ds = downsample(x_img, y_img, target_samples)
        x_audio_ds, y_audio_ds = x_audio, y_audio
    else:
        x_audio_ds, y_audio_ds = downsample(x_audio, y_audio, target_samples)
        x_img_ds, y_img_ds = x_img, y_img

    return x_img_ds, x_audio_ds, y_img_ds, y_audio_ds

