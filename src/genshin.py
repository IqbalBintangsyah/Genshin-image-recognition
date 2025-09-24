import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

import kagglehub

dataset_url = "/app/dataset/"
data_dir = pathlib.Path(dataset_url).with_suffix('')

image_count = len(list(data_dir.glob('*/*.jpg')))

batch_size = 20
img_height = 400
img_width = 400

with tf.device('/cpu:0'):
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size
	)
	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size
	)

	normalization_layer = tf.keras.layers.Rescaling(1./255)
	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	image_batch, label_batch = next(iter(normalized_ds))

	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	num_classes = 5

	model = tf.keras.Sequential([
		tf.keras.layers.Rescaling(1./255),
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(num_classes)
	])

	model.compile(
		optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=3
	)