#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import librosa
import gc
import seaborn as sns

from shared.utils import *
from shared.local_path import *

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#%% md
# # Data Preprocessing
#  ## Audio Data
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
del filepath, label
audio.head()
#%%
le = LabelEncoder()
le.fit(audio['label'])
audio['label_encoded'] = le.transform(audio['label'])
num_classes = len(le.classes_)
#%%
audio.label_encoded.unique()
#%%
with tf.device('/GPU:0'):
    audio['data'] = audio.filepath.apply(preprocess_audio)
#%%
x_audio = np.stack(audio['data'].values)
y_audio = audio['label_encoded'].values
del audio
#%%
x_audio_train, x_temp, y_audio_train, y_temp = train_test_split(
    x_audio, y_audio, test_size=0.3, random_state=42, stratify=y_audio
)

x_audio_val, x_audio_test, y_audio_val, y_audio_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

del x_audio, y_audio
#%%
x_audio_train.shape[0] == len(y_audio_train)
#%% md
# ## Image Data
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
    train_df['data'] = train_df['filepath'].apply(preprocess_image)
    img_test_df['data'] = img_test_df['filepath'].apply(preprocess_image)
#%%
gc.collect()
tf.keras.backend.clear_session()
#%%
x_img = np.stack(train_df['data'].values)
y_img = train_df['label_encoded'].values

x_img_test = np.stack(img_test_df['data'].values)
y_img_test = img_test_df['label_encoded'].values

del train_df, img_test_df
x_img_train, x_img_val, y_img_train, y_img_val = train_test_split(x_img, y_img, test_size=0.2, random_state=42, stratify=y_img)
del x_img, y_img
#%% md
# # Modeling
# ## Set Base Model InceptionV3
#%%
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
#%%
input_image = tf.keras.layers.Input(shape=(128, 110, 3), name='input_image')
input_audio = tf.keras.layers.Input(shape=(128, 110, 3), name='input_audio')

inputs = tf.keras.layers.Concatenate(axis=1)([input_image, input_audio])

x1 = base_model(inputs)
flat = tf.keras.layers.Flatten()(x1)
fc1 = tf.keras.layers.Dense(512, activation='relu')(flat)
fc2 = tf.keras.layers.Dense(512, activation='relu')(fc1)

out1 = tf.keras.layers.Dense(7, activation='softmax', name='image_class')(fc2)
out2 = tf.keras.layers.Dense(8, activation='softmax', name='audio_class')(fc2)

model = tf.keras.models.Model(inputs=[input_image, input_audio], outputs=[out1, out2])
#%%
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
#%%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.),
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
min_samples = min(len(x_img_train), len(x_audio_train), len(y_img_train), len(y_audio_train))

# Filter inputs and labels
x_img_train = x_img_train[:min_samples]
x_audio_train = x_audio_train[:min_samples]
y_img_train = y_img_train[:min_samples]
y_audio_train = y_audio_train[:min_samples]

# Print shapes after filtering
print("Input Image Shape:", x_img_train.shape) 
print("Input Audio Shape:", x_audio_train.shape)  
print("Image Labels Shape:", y_img_train.shape) 
print("Audio Labels Shape:", y_audio_train.shape)
#%%
print("Input Image Type:", x_img_train.dtype) 
print("Input Audio Type:", x_audio_train.dtype)  
print("Image Labels Type:", y_img_train.dtype) 
print("Audio Labels Type:", y_audio_train.dtype)
#%%
min_samples = min(len(x_img_val), len(x_audio_val), len(y_img_val), len(y_audio_val))

x_img_val = x_img_val[:min_samples]
x_audio_val = x_audio_val[:min_samples]
y_img_val = y_img_val[:min_samples]
y_audio_val = y_audio_val[:min_samples]
#%%
class EarlyStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') < .5:
            self.model.stop_training = True
            
history = model.fit(
    x=[x_img_train, x_audio_train],
    y=[y_img_train, y_audio_train], 
    validation_data=(
        [x_img_val, x_audio_val], 
        [y_img_val, y_audio_val]  
    ),
    callbacks=EarlyStop(),
    epochs=50, 
    batch_size=32  
)
model.save('model_sc1t3.h5')
#%%
def plot_and_save(metric_name, history, ylabel, filename):
    plt.figure()
    plt.plot(history.history[metric_name], label=f'Train {ylabel}')
    if f'val_{metric_name}' in history.history:
        plt.plot(history.history[f'val_{metric_name}'], label=f'Val {ylabel}')
    plt.title(f'{ylabel} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
#%%
plot_and_save('loss', history, 'Loss', 'loss_plot.png')
plot_and_save('image_class_loss', history, 'Image Classification Loss', 'image_class_loss_plot.png')
plot_and_save('audio_class_loss', history, 'Audio Classification Loss', 'audio_class_loss_plot.png')
plot_and_save('image_class_accuracy', history, 'Image Classification Accuracy', 'image_class_accuracy_plot.png')
plot_and_save('audio_class_accuracy', history, 'Audio Classification Accuracy', 'audio_class_accuracy_plot.png')
#%% md
# # Evaluation
# ## Make sure the data has same sample amount
#%%
min_samples = min(len(x_img_test), len(x_audio_test), len(y_img_test), len(y_audio_test))

x_img_test = x_img_test[:min_samples]
x_audio_test = x_audio_test[:min_samples]
y_img_test = y_img_test[:min_samples]
y_audio_test = y_audio_test[:min_samples]
#%% md
# ## Use the model.evaluate
#%%
model.evaluate(
    x=[x_img_test, x_audio_test],
    y=[y_img_test, y_audio_test],
    batch_size=32,
)
#%% md
# ## Use confusion matrix
# ### Create the model prediction first berfore the cm
#%%
predictions = model.predict(
    x=[x_img_test, x_audio_test],
    batch_size=32,
)

image_preds = predictions[0]
audio_preds = predictions[1]  
#%%
image_class = np.argmax(predictions[0], axis=1)
audio_class = np.argmax(predictions[1], axis=1)

image_cm = confusion_matrix(y_img_test, image_class)
audio_cm = confusion_matrix(y_audio_test, audio_class)
#%%
plt.figure(figsize=(8, 6))
sns.heatmap(image_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Image Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('image_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#%%
plt.figure(figsize=(8, 6))
sns.heatmap(audio_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Audio Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('audio_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
