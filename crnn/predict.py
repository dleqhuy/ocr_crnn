import argparse
from pathlib import Path

import glob
import yaml
import tensorflow as tf

from tensorflow import keras
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from dataset_factory import DatasetBuilder
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--weight', type=str, required=True, default='',
                    help='The saved weight path.')
parser.add_argument('--post', type=str, help='Post processing.')
parser.add_argument('--images', type=str, required=True, 
                    help='Image file or folder path.')

args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']


model = tf.saved_model.load(args.weight)


if args.post == 'greedy':
    postprocess = CTCGreedyDecoder(config['vocab_path'])
elif args.post == 'beam_search':
    postprocess = CTCBeamSearchDecoder(config['vocab_path'])
else:
    postprocess = None


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def read_img_and_resize(path, img_width, img_height, channel):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channel)

    img = distortion_free_resize(img, (img_width, img_height))
    img = tf.cast(img, tf.float32) / 255.0
    return img

img_paths = glob.glob(f'{args.images}/*.jpg')

for img_path in img_paths:
    img = read_img_and_resize(str(img_path), config['img_width'],  config['img_height'], config['channel'])
    img = tf.expand_dims(img, 0)
    outputs = model(img)
    
    y_pred, probability = postprocess.call(outputs)
    
    print(f'Path: {img_path}, y_pred: {y_pred.numpy().astype(str)}, '
          f'probability: {probability.numpy()}')