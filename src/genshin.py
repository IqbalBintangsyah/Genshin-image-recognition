import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

import kagglehub

# Download latest version
path = kagglehub.dataset_download("just1ce5/genshin-impact-characters-dataset")

print("Path to dataset files:", path)