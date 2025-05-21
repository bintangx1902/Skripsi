import cv2
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

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


def preprocess_image(img_path, dsize: tuple = (128, 128), normalize=False, augment=False, rescale: float = None):
    """
    :param augment: boolean, if u need to augment the image 
    :param normalize: boolean, if u need to normalize with (feature-mean)/std
    :param rescale: float, the feature will be rescaled by this value
    :param img_path: path to image (full path or can be absolute path)  
    :param dsize: new data size, default is 100 x 100 px
    :return: upsampled image to 100x100 pixels
    """
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

    if augment:
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        factor = 1.0 + 0.1 * (np.random.rand() - 0.5)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

    if rescale is not None:
        image = np.array(image) * rescale
    return image


def plot_model(model: tf.keras.Model, show_shapes=False, show_layer_names=False):
    """
    :param model: the model you want to plot
    :param show_shapes: if you want to show the shape each layer of the model
    :param show_layer_names: if you want to show the layer names each layer of the model
    :return: 
    """
    return tf.keras.utils.plot_model(model, show_shapes=show_shapes, show_layer_names=show_layer_names)


def stratified_downsample(x_img, x_audio, y_img, y_audio):
    """
    Downsamples the image and audio datasets while ensuring each class is preserved
    and the number of samples matches the smallest class size in both datasets.
    It also ensures that x_img[i] corresponds to y_img[i] and x_audio[i] corresponds to y_audio[i].

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

    # Determine the minimum number of samples to match the smallest dataset size
    min_samples = min(len(x_img), len(x_audio))

    # Prepare lists to store downsampled indices
    selected_indices_img = []
    selected_indices_audio = []

    # Downsample image data to match the smallest class size in both datasets
    for cls in np.unique(y_img):
        # Find indices for this class in image data
        indices_img = np.where(y_img == cls)[0]
        # Shuffle and select the minimum number of samples for the class
        np.random.shuffle(indices_img)
        selected_indices_img.extend(indices_img[:min_samples])

    # Downsample audio data to match the smallest class size in both datasets
    for cls in np.unique(y_audio):
        # Find indices for this class in audio data
        indices_audio = np.where(y_audio == cls)[0]
        # Shuffle and select the minimum number of samples for the class
        np.random.shuffle(indices_audio)
        selected_indices_audio.extend(indices_audio[:min_samples])

    # Convert to numpy array for indexing
    selected_indices_img = np.array(selected_indices_img)
    selected_indices_audio = np.array(selected_indices_audio)

    # Sort the indices so that the data and labels remain consistent
    selected_indices_img = np.sort(selected_indices_img)
    selected_indices_audio = np.sort(selected_indices_audio)

    # Downsample both image and audio data using the selected indices
    x_img_ds = x_img[selected_indices_img]
    y_img_ds = y_img[selected_indices_img]
    x_audio_ds = x_audio[selected_indices_audio]
    y_audio_ds = y_audio[selected_indices_audio]

    return x_img_ds, x_audio_ds, y_img_ds, y_audio_ds


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

    del n_img, n_audio, target_samples
    return x_img_ds, x_audio_ds, y_img_ds, y_audio_ds


def balance_dataframe(df, target_total):
    labels = df['label'].unique()
    num_labels = len(labels)
    samples_per_label = target_total // num_labels

    balanced_parts = []
    for label in labels:
        df_label = df[df['label'] == label]
        sample_n = min(samples_per_label, len(df_label))
        balanced_parts.append(df_label.sample(n=sample_n, random_state=42))

    balanced_df = pd.concat(balanced_parts)

    # If we're short due to label imbalance, fill the rest randomly
    if len(balanced_df) < target_total:
        remaining = target_total - len(balanced_df)
        available_df = df.drop(balanced_df.index)
        if not available_df.empty:
            filler = available_df.sample(n=min(remaining, len(available_df)), random_state=42)
            balanced_df = pd.concat([balanced_df, filler])

    return balanced_df.reset_index(drop=True)


def downsample_df(df1, df2):
    target = min(len(df1), len(df2))
    new_df1 = balance_dataframe(df1, target)
    new_df2 = balance_dataframe(df2, target)
    return new_df1, new_df2
