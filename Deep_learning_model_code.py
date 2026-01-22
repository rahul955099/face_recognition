import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 20

DATA_DIR = os.path.join(BASE_DIR, "dataset", "preprocessed")
MODEL_PATH = os.path.join(BASE_DIR, "face_recognition_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.txt")



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)



AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(100).cache().prefetch(AUTOTUNE)#
val_ds = val_ds.cache().prefetch(AUTOTUNE)#




data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])



base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # IMPORTANT for small dataset



model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
#Prevents losing best model if training fluctuates
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)


model.save(MODEL_PATH)

with open(CLASS_PATH, "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("✔ Model saved to:", MODEL_PATH)
print("✔ Class names saved to:", CLASS_PATH)
