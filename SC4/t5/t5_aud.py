# %%
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from shared.colab_path import *
from shared.utils import *

# %% md
# # Data Preprocessing
# ## Audio Data
# %%
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
# %%
le = LabelEncoder()
le.fit(audio['label'])
audio['label_encoded'] = le.transform(audio['label'])
num_classes = len(le.classes_)
# %%
with tf.device('/GPU:0'):
    audio['data'] = audio.filepath.apply(preprocess_audio)
# %%
# Audio Data
audio = audio.sample(frac=1).reset_index(drop=True)
_, aud_temp = train_test_split(audio, test_size=0.5, random_state=42)
aud_val_df, aud_test_df = train_test_split(aud_temp, test_size=.5, random_state=42)
# %%
x_audio_train = np.stack(audio['data'].values)
y_audio_train = np.array(audio.label_encoded.values)

x_audio_val = np.stack(aud_val_df['data'].values)
y_audio_val = np.array(aud_val_df.label_encoded.values)

x_audio_test = np.stack(aud_test_df['data'].values)
y_audio_test = np.array(aud_test_df.label_encoded.values)

del _, aud_temp, aud_test_df, aud_val_df, audio
# %%
print(x_audio_train.shape)
print(x_audio_val.shape)
print(x_audio_test.shape)
# %% md
# # Modeling
# ## Model Image
# %%
base_model = tf.keras.applications.VGG19(
    weights='imagenet',
    include_top=False,
    input_tensor=tf.keras.layers.Input(shape=(128, 110, 3)),
    pooling="avg",
)

for layer in base_model.layers:
    layer.trainable = False
# %%
with tf.device('/CPU:0'):
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
# %%
plot_model(model, show_shapes=True)
# %%
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
                                                    factor=0.2,
                                                    min_lr=1e-6,
                                                    patience=3,
                                                    mode='min'
                                                    )
# %%
with tf.device('/GPU:0'):
    history = model.fit(
        x_audio_train,
        y_audio_train,
        validation_data=(x_audio_val, y_audio_val),
        callbacks=[lr_scheduler],
        epochs=100,
        verbose=2,
        batch_size=BATCH_SIZE,
        steps_per_epoch=len(x_audio_train) // BATCH_SIZE,
        validation_steps=len(x_audio_val) // BATCH_SIZE,
    )

# %% md
# # Evaluate Model
# ## Using the model.evaluate
# ### using test set
# %%
print(model.evaluate(x_audio_test, y_audio_test, batch_size=BATCH_SIZE, steps=len(x_audio_test) // BATCH_SIZE))
# %% md
# ### Using validation set
# %%
print(model.evaluate(x_audio_val, y_audio_val, batch_size=BATCH_SIZE, steps=len(x_audio_val) // BATCH_SIZE))


# %%
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


# %%
plot_and_save('loss', history, 'Audio_Loss', 'Audio_loss_plot-nolr.png')
plot_and_save('accuracy', history, 'Audio Classification Accuracy', 'Audio_class_accuracy_plot-nolr.png')
# %%
predictions = model.predict(x_audio_test)
predictions = np.argmax(predictions, axis=1)
# %%
classes = [x for x in os.listdir(AUDIO_PATH)]
print(classes)
# %%
cm = confusion_matrix(y_audio_test, predictions)

plt.figure(figsize=(10, 8))  # Set the figure size if needed
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

plt.title('Confusion Matrix Audio', pad=20, fontsize=20, fontweight="bold")
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Set ticks for the x and y axes using class names
plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
plt.yticks(ticks=range(len(classes)), labels=classes, rotation=0)
plt.savefig('confusion_matrix_audio.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
print(classification_report(predictions, y_audio_test))
# %%
model.save('model_sc4_t3_audio.h5')
