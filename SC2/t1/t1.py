#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from shared.utils import *
from shared.local_path import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#%% md
# # Data Preprocessing
# ## Audio data
#%%
label = []
filepath = []

for classes in os.listdir(AUDIO_PATH):
    for file in os.listdir(os.path.join(AUDIO_PATH, classes)):
        file_path = os.path.join(AUDIO_PATH, classes, file)

        filepath.append(file_path)
        label.append(classes)

audio = pd.DataFrame({
    'filepath': filepath,
    'label': label,
})

audio.head()
#%%
le = LabelEncoder()
le.fit(audio['label'])
audio['label_encoded'] = le.transform(audio['label'])
num_classes = len(le.classes_)
#%%
with tf.device('/GPU:0'):
    audio['data'] = audio.filepath.apply(preprocess_audio)
#%%
del filepath, label
#%%
x_audio = np.stack(audio['data'].values)
y_audio = audio['label_encoded'].values
print(x_audio.shape)
del audio
#%%
x_audio_train, x_temp, y_audio_train, y_temp = train_test_split(
    x_audio, y_audio, test_size=0.4, random_state=42, stratify=y_audio
)

x_audio_val, x_audio_test, y_audio_val, y_audio_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

del x_audio, y_audio
print(x_audio_train.shape)
#%%
x_audio_train.shape[0] == len(y_audio_train)
#%% md
# ## Image data
#%%
filepath = []
label = []

i = 0
for classes in os.listdir(TRAIN_IMAGE_PATH):
    for file in os.listdir(os.path.join(TRAIN_IMAGE_PATH, classes)):
        path = os.path.join(TRAIN_IMAGE_PATH, classes, file)
        filepath.append(path)
        label.append(classes)

train_df = pd.DataFrame({
    'filepath': filepath,
    'label': label
})

del filepath, label

print(train_df.shape)
print(train_df['label'].unique())
train_df.head()
#%%
filepath = []
label = []

i = 0
for classes in os.listdir(TEST_IMAGE_PATH):
    for file in os.listdir(os.path.join(TEST_IMAGE_PATH, classes)):
        path = os.path.join(TEST_IMAGE_PATH, classes, file)
        filepath.append(path)
        label.append(classes)

img_test_df = pd.DataFrame({
    'filepath': filepath,
    'label': label
})

del filepath, label

print(img_test_df.shape)
print(img_test_df['label'].unique())
img_test_df.head()
#%%
le = LabelEncoder()
le.fit(train_df['label'])
train_df['label_encoded'] = le.transform(train_df['label'])
img_test_df['label_encoded'] = le.transform(img_test_df['label'])
#%%
with tf.device('/GPU:0'):
    train_df['data'] = train_df['filepath'].apply(lambda x : preprocess_image(x, (100, 100)))
    img_test_df['data'] = img_test_df['filepath'].apply(lambda x : preprocess_image(x, (100, 100)))
#%%
train_df.shape
#%%
x_img = np.stack(train_df['data'].values)
y_img = train_df['label_encoded'].values

x_img_test = np.stack(img_test_df['data'].values)
y_img_test = img_test_df['label_encoded'].values

del train_df, img_test_df
#%%
x_img.shape[1:] == x_img_test.shape[1:]
#%%
x_img.shape
#%% md
# # Modeling
#%%
def create_base_model(instance_name):
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        pooling='max',
        weights='imagenet',
        input_shape=(96, 96, 3)
    )
    inputs = tf.keras.Input(shape=(96, 96, 3))
    outputs = base(inputs)
    return tf.keras.Model(inputs, outputs, name=f"inception_v3_{instance_name}")
#%%
input_image = tf.keras.layers.Input(shape=(96, 96, 3), name='input_image')
input_audio = tf.keras.layers.Input(shape=(96, 96, 3), name='input_audio')

image_features = create_base_model('image')(input_image)
audio_features = create_base_model('audio')(input_audio)

feature = tf.keras.layers.Concatenate()([image_features, audio_features])

fc1 = tf.keras.layers.Dense(512, activation='relu')(feature)
fc2 = tf.keras.layers.Dense(512, activation='relu')(fc1)

out1 = tf.keras.layers.Dense(8, activation='softmax', name='image_class')(fc2)
out2 = tf.keras.layers.Dense(7, activation='softmax', name='audio_class')(fc2)

model = tf.keras.models.Model(inputs=[input_image, input_audio], outputs=[out1, out2])
#%%
tf.keras.utils.plot_model(model, show_shapes=True)
#%%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'image_class': 'sparse_categorical_crossentropy',
        'audio_class': 'sparse_categorical_crossentropy'
    },
    metrics={
        'image_class': 'accuracy',
        'audio_class': 'accuracy' 
    }
)
#%%
min_samples = min(len(x_img), len(x_audio_train), len(y_img), len(y_audio_train))

# Filter inputs and labels
x_img = x_img[:min_samples]
x_audio_train = x_audio_train[:min_samples]
y_img = y_img[:min_samples]
y_audio_train = y_audio_train[:min_samples]

# Print shapes after filtering
print("Input Image Shape:", x_img.shape) 
print("Input Audio Shape:", x_audio_train.shape)  
print("Image Labels Shape:", y_img.shape) 
print("Audio Labels Shape:", y_audio_train.shape)
#%%
print("Input Image Type:", x_img.dtype) 
print("Input Audio Type:", x_audio_train.dtype)  
print("Image Labels Type:", y_img.dtype) 
print("Audio Labels Type:", y_audio_train.dtype)
#%%
with tf.device('/GPU:0'):
    history = model.fit(
        x=[x_img, x_audio_train],
        y=[y_img, y_audio_train], 
        validation_data=(
            [x_img_test, x_audio_val], 
            [y_img_test, y_audio_val]  
        ),
        epochs=10, 
        batch_size=32  
    )

#%%
