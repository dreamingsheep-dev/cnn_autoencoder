import tensorflow as tf


class CNNAutoencoder(object):

    def __init__(self, img_height, img_width, img_depth, mode, learning_rate=1e-3, name=None):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.mode = mode
        self.learning_rate = learning_rate
        self.name = name if name else 'CNNAutoencoder'

        self.n_classes=10

        self.step = 0

    def build_graph(self):

        self._build_model()

        if self.mode == 'train':
            self._build_train_op()

    def _build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_model(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.img_height, self.img_width, self.img_depth))
        print('build_model!')
        print(self.X)
        # Encoder
        self.x = tf.layers.conv2d(self.X, filters=16, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv1' % self.name)
        print(self.x)
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='%s_pool1' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv2' % self.name)
        print(self.x)
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='%s_pool2' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d(self.x, filters=1, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv3' % self.name)
        print(self.x)
        self.x=tf.layers.flatten(self.x)
        print(self.x)
        self.encoded= tf.layers.dense(self.x, 10, activation=tf.nn.sigmoid)
        #self.encoded = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='%s_encoded' % self.name)
        self.x=tf.layers.dense(self.encoded,18*18,activation=tf.nn.relu)
        self.x=tf.reshape(self.x,[-1,18,18,1])
        # Decoder
        self.x = tf.layers.conv2d(self.x, filters=1, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv4' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d_transpose(self.x, filters=8, kernel_size=(2, 2), strides=(2, 2), name='%s_deconv4' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv5' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d_transpose(self.x, filters=8, kernel_size=(2, 2), strides=(2, 2), name='%s_deconv5' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d(self.x, filters=16, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME',name='%s_conv6' % self.name)
        print(self.x)
        self.x = tf.layers.conv2d_transpose(self.x, filters=16, kernel_size=(2, 2), strides=(2, 2), name='%s_deconv6' % self.name)
        print(self.x)
        self.recon = tf.layers.conv2d(self.x, filters=1, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_recon' % self.name)
        print(self.recon)
        # Use mean-square-error as loss
        self.loss = tf.reduce_mean(tf.square(self.X - self.recon))

        self.loss_summary = tf.summary.scalar("Loss", self.loss)

    def get_reconstructed_images(self, sess, X):

        feed_dict = {
            self.X: X
        }

        return sess.run(self.recon, feed_dict=feed_dict)
