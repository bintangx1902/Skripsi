#%%
import gc

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from shared.colab_path import *
from shared.utils import *
#%% md
# ### Read Image Data
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
#%% md
# ## transform the filepath into image data
#%%
with tf.device('/GPU:0'):
    train_df['data'] = train_df['filepath'].apply(
        lambda x: preprocess_image(x, (100, 100), normalize=True, augment=True))
    img_test_df['data'] = img_test_df['filepath'].apply(
        lambda x: preprocess_image(x, (100, 100), normalize=True)
    )

#%% md
# ## Split and Shuffle Dataset into train, val, and test
#%%
# Image Data
# train_df = train_df.sample(frac=1, random_state=19).reset_index(drop=True)
train_df, img_val_df = train_test_split(train_df, test_size=.2, random_state=100, shuffle=True)
#%% md
# ## Downsampling with following least dataset amount
#%%
# train_df, audio = downsample_df(train_df, audio)
# img_val_df, aud_val_df = downsample_df(img_val_df, aud_val_df)
# img_test_df, aud_test_df = downsample_df(img_test_df, aud_test_df)
#%%
# print(len(train_df))
# print(len(img_val_df))
# print(len(img_test_df))
# 
# print("===============================")
# 
# print(len(audio))
# print(len(aud_val_df))
# print(len(aud_test_df))
#%% md
# ## Convert to Numpy
#%%
# # Audio Data
# x_audio_train = np.stack(audio['data'].values)
# y_audio_train = np.array(audio['label_encoded'].values)
# 
# x_audio_val = np.stack(aud_val_df['data'].values)
# y_audio_val = np.array(aud_val_df['label_encoded'].values)
# 
# x_audio_test = np.stack(aud_test_df['data'].values)
# y_audio_test = np.array(aud_test_df['label_encoded'].values)
#%%
x_img_train = np.stack(train_df['data'].values)
y_img_train = np.array(train_df['label_encoded'].values)

x_img_val = np.stack(img_val_df['data'].values)
y_img_val = np.array(img_val_df['label_encoded'].values)

x_img_test = np.stack(img_test_df['data'].values)
y_img_test = np.array(img_test_df['label_encoded'].values)
#%%
del train_df, img_val_df, img_test_df
#%% md
# # Modeling
# ## Creating Model
#%%
base = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(100, 100, 3),
    pooling=None
)

for layer in base.layers:
    layer.trainable = False

#%%
input_image = tf.keras.Input(shape=(100, 100, 3), name='input_image')

left = base(input_image)
left = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1)(left)
left = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(left)
left = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-3)(left)
left = tf.keras.layers.Dropout(0.3)(left)
left = tf.keras.layers.Flatten()(left)
left = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(left)
out1 = tf.keras.layers.Dense(7, activation='softmax', name='image_class')(left)

model = tf.keras.models.Model(input_image, out1)
#%%
plot_model(model, show_shapes=True, show_layer_names=True)
#%% md
# ## Compile the model
#%%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics='accuracy',
)
#%%
tf.keras.backend.clear_session()
gc.collect()
#%%
early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    mode='min',
    # start_from_epoch=2,
    patience=5
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('model', 'best_model.keras'),
    verbose=2,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1,
                                                    min_lr=1e-6,
                                                    patience=3,
                                                    mode='min'
                                                    )
#%% md
# ## Train the model
#%%
gc.collect()
tf.keras.backend.clear_session()
#%%
history = model.fit(
    x=x_img_train,
    y=y_img_train,
    validation_data=(
        x_img_val,
        y_img_val
    ),
    callbacks=[lr_scheduler],
    epochs=50,
    batch_size=BATCH_SIZE,
    steps_per_epoch=len(x_img_train) // BATCH_SIZE,
    validation_steps=len(x_img_val) // BATCH_SIZE,
)

#%% md
# ## Plot the training result
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
plot_and_save('loss', history, 'Loss', 'image_loss_plot.png')
plot_and_save('accuracy', history, 'Image Classification Accuracy', 'image_class_accuracy_plot.png')
#%% md
# # Evaluation
# ## Using model.evaluate
#%%
gc.collect()
tf.keras.backend.clear_session()
#%%
model.evaluate(x_img_val, y_img_val, batch_size=BATCH_SIZE, verbose=2,
               steps=len(x_img_val) // BATCH_SIZE)
#%%
model.evaluate(x_img_test, y_img_test, batch_size=BATCH_SIZE, verbose=2,
               steps=len(x_img_test) // BATCH_SIZE)
#%% md
# ## Using confusion matrix
#%%
predictions = model.predict(x_img_test)
predictions = np.argmax(predictions, axis=1)

classes = [x for x in os.listdir(TRAIN_IMAGE_PATH)]
print(classes)
#%% md
# ### Plot the confusion Matrix
#%%
cm = confusion_matrix(y_img_test, predictions)
plt.figure(figsize=(10, 8))  # Set the figure size if needed
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

plt.title('Confusion Matrix Image', pad=20, fontsize=20, fontweight="bold")
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
plt.yticks(ticks=range(len(classes)), labels=classes, rotation=0)
plt.savefig('confusion_matrix_image.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
# plt.figure(figsize=(10, 8))  # Set the figure size if needed
# sns.heatmap(audio_cm, annot=True, cmap='Blues', fmt='g')
# 
# plt.title('Confusion Matrix Audio', pad=20, fontsize=20, fontweight="bold")
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# 
# plt.xticks(ticks=range(len(aud_classes)), labels=aud_classes, rotation=45)
# plt.yticks(ticks=range(len(aud_classes)), labels=aud_classes, rotation=0)
# plt.savefig('confusion_matrix_audio.png', dpi=300, bbox_inches='tight')
# plt.show()
#%%
print(classification_report(y_img_test, predictions))
#%%
# print(classification_report(y_audio_test, audio_class))
#%%
model.save('model_sc4t5_img.h5')
#%%

#%%
