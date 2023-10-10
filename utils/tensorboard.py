import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
class TensorBoard(object):
    def __init__(self, log_dir):
        "Create a summary writer logging to log_dir.“”"
        self.writer = tf.summary.create_file_writer(log_dir)
    def scalar_summary(self, tag, value, step):
        "Log a scalar variable.“”"
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        # self.writer.add_summary(summary, step)
    def image_summary(self, tag, images, step):
        "Log a list of images."
        img_data = []
        for i, img in enumerate(images):
            try:
                s = StringIO()
            except:
                s = BytesIO()
            Image.fromarray(img).save(s, format="png")
            img_data.append(img)
        # np.reshape(i)
        with self.writer.as_default():
            tf.summary.image(name=tag, data= np.reshape(img_data, (-1, 288, 800, 3)), step=step)
            self.writer.flush()
    def histo_summary(self, tag, values, step, bins=1000):
        "Log a histogram of the tensor of values."
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
            self.writer.flush()