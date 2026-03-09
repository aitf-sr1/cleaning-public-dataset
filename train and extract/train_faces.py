import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_labels(csv_path: str) -> pd.DataFrame:
    """Read a label CSV and normalize the ClipID column without the .avi extension."""
    df = pd.read_csv(csv_path)
    # remove any whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    # normalize clip id (folder name) by stripping extensions
    df['ClipID'] = df['ClipID'].astype(str).str.replace(r"\.avi$", "", regex=True)
    return df


def create_dataset(faces_root: str,
                   labels_df: pd.DataFrame,
                   label_columns: list = None,
                   batch_size: int = 32,
                   image_size=(224, 224)) -> tf.data.Dataset:
    """Build a tf.data dataset from a directory of cropped face folders.

    Each subdirectory of ``faces_root`` represents one clip and
    contains a small number of images (typically 4).  All of the images
    in a given clip share the same set of emotion labels, which are
    looked up in ``labels_df``.  ``labels_df`` must contain a column
    named ``ClipID`` that matches the folder names, plus the emotion
    columns (e.g. Boredom, Engagement, Confusion, Frustration).
    """

    if label_columns is None:
        label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

    filepaths = []
    labels = []

    for subject in os.listdir(faces_root):
        subject_dir = os.path.join(faces_root, subject)
        if not os.path.isdir(subject_dir):
            continue
        for clip in os.listdir(subject_dir):
            clip_dir = os.path.join(subject_dir, clip)
            if not os.path.isdir(clip_dir):
                continue
            # look up label row
            row = labels_df[labels_df['ClipID'] == clip]
            if row.empty:
                # missing label, skip
                continue
            # multi-output: take all requested columns
            label_values = row[label_columns].values[0].astype(float)
            for fname in os.listdir(clip_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(clip_dir, fname))
                    labels.append(label_values)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)

        # data augmentation before preprocessing
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

        # use EfficientNet preprocessing after augmentation
        img = preprocess_input(img)

        return img, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # cache processed images for faster training
    ds = ds.shuffle(buffer_size=5000)  # larger shuffle buffer
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(output_dim: int, fine_tune: bool = False) -> tf.keras.Model:
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    inp = layers.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # improved head network
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(inp, out)

    if fine_tune:
        # unfreeze last 20 layers for fine-tuning
        for layer in base.layers[:-20]:
            layer.trainable = False
        for layer in base.layers[-20:]:
            layer.trainable = True

    return model


def main():
    # default paths relative to script
    train_csv = os.path.join('Labels', 'TrainLabels.csv')
    val_csv = os.path.join('Labels', 'ValidationLabels.csv')
    faces_root = os.path.join('FacesCropped')

    # load labels
    train_labels = load_labels(train_csv)
    val_labels = load_labels(val_csv)

    # columns we want to predict (multi-output)
    label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

    train_ds = create_dataset(os.path.join(faces_root, 'Train'), train_labels, label_columns)
    val_ds = create_dataset(os.path.join(faces_root, 'Validation'), val_labels, label_columns)

    output_dim = len(label_columns)

    # initial training with frozen base
    model = build_model(output_dim, fine_tune=False)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()

    print("Initial training with frozen backbone...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5,
              callbacks=callbacks)

    # fine-tuning with unfrozen layers
    print("Unfreezing backbone for fine-tuning...")

    base_model = model.layers[1]  # EfficientNet base
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # even lower LR
                  loss='mse',
                  metrics=['mae'])

    print("Fine-tuning with unfrozen backbone...")
    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10,
              callbacks=callbacks_ft)

    model.save('efficientnet_b0_facebook.h5')


if __name__ == '__main__':
    main()
