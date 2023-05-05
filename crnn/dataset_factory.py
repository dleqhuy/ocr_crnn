import tensorflow as tf

from tensorflow.keras import layers

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    # tf < 2.4.0
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetBuilder:

    def __init__(self, vocab_path, img_width, img_height, channel):
        with open(vocab_path) as file:
            vocab = [line.rstrip() for line in file]
        self.char_to_num = layers.StringLookup(
            vocabulary=vocab, mask_token=None,
        )

        self.img_width = img_width
        self.img_height = img_height
        self.channel = channel
    @property
    def num_classes(self):
        return len(self.char_to_num.get_vocabulary())

    def _distortion_free_resize(self, image, img_size):
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

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_png(img, channels=self.channel)
        img = self._distortion_free_resize(img, (self.img_width, self.img_height))
        img = tf.cast(img, tf.float32) / 255.0
                
        return img, label
    
    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.char_to_num, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return imgs, tokens
    
    def __call__(self, dataframe, batch_size, cache=False, shuffle=False, drop_remainder=False):

        ds = tf.data.Dataset.from_tensor_slices((dataframe['file_path'], dataframe['label']))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=500)
            
        ds = ds.map(self._decode_img, AUTOTUNE)
        
        if cache:
            ds = ds.cache()
            
        ds = ds.padded_batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds