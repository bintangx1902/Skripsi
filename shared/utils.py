import random

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


def preprocess_image(img_path, dsize: tuple = (128, 128), normalize=False, augment=False, 
                     rescale: float = None, preprocess_function=None):
    """
    Preprocesses an image with optional normalization, rescaling, augmentation, or custom preprocessing.

    Args:
        img_path (str): Path to the image (relative or absolute).
        dsize (tuple): Target size for resizing (width, height), default is (128, 128).
        normalize (bool): If True, normalize using ImageNet mean and std.
        augment (bool): If True, apply one random augmentation (flip, rotate, or zoom).
        rescale (float, optional): If provided, rescale image pixel values by this factor.
        preprocess_function (callable, optional): Custom preprocessing function, e.g., tf.keras.applications.vgg19.preprocess_input.

    Returns:
        np.ndarray or tf.Tensor: Processed image as a NumPy array or TensorFlow tensor (if preprocess_function is used).

    Raises:
        FileNotFoundError: If the image path is invalid.
        ValueError: If input parameters are invalid (e.g., negative dsize, conflicting options).
    """
    if preprocess_function is not None and not callable(preprocess_function):
        raise ValueError("preprocess_function must be a callable function.")
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

    if normalize and rescale is None:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

    if rescale is not None and not normalize:
        image = np.array(image) * rescale

    if augment:
        # Choose one augmentation randomly
        augmentations = ['flip_horizontal', 'flip_vertical', 'rotate',]
        chosen_aug = random.choice(augmentations)

        if chosen_aug == 'flip_horizontal':
            image = cv2.flip(image, 1)
        elif chosen_aug == 'flip_vertical':
            image = cv2.flip(image, 0)
        elif chosen_aug == 'rotate':
            # Random rotation between -30 and 30 degrees
            angle = random.uniform(-30, 30)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        elif chosen_aug == 'zoom':
            # Random zoom between 0.8x and 1.2x
            zoom_factor = random.uniform(0.8, 1.2)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            if zoom_factor > 1.0:
                # Zoom in: crop the center
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = resized[start_y:start_y + h, start_x:start_x + w]
            else:
                # Zoom out: pad with replicated borders
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                image = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x,
                                           cv2.BORDER_REPLICATE)

    if preprocess_function is not None:
        return np.array(preprocess_function(image))
    return image


def preprocess_image_tf(img_path, dsize: tuple = (128, 128), decode=False, cv=False, normalize=False, augment=False,
                        rescale: float = None):
    if decode:
        image = tf.io.read_file(img_path)
        image = tf.io.decode_image(image, expand_animations=False, dtype=tf.float32, channels=3)
    else:
        image = tf.keras.utils.load_img(img_path)
        image = tf.keras.preprocessing.image.img_to_array(image)
        if image.shape[-1] == 1:  # grayscale
            image = tf.image.grayscale_to_rgb(image)
        image = tf.convert_to_tensor(image, dtype=tf.float32)  # Ensure it's a tensor

    if cv:
        image_np = image.numpy() if isinstance(image, tf.Tensor) else image
        image_np = cv2.resize(image_np, dsize, interpolation=cv2.INTER_LANCZOS4)
        image = tf.convert_to_tensor(image_np, dtype=tf.float32)
    else:
        image = tf.image.resize(image, dsize, method='lanczos5')

    if normalize:
        image = image / 255.0

    if rescale is not None and not normalize:
        image = image * rescale

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


def downsample_df(df1, df2):
    target = min(len(df1), len(df2))
    new_df1 = balance_dataframe(df1, target)
    new_df2 = balance_dataframe(df2, target)
    return new_df1, new_df2


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
