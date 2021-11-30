import tensorflow as tf

INPUT_HEIGHT = 200
INPUT_WIDTH = 200


class FingNet(tf.keras.Model):
    def __init__(self, alpha, lmbda, d_latent):
        super(FingNet, self).__init__()
        self.alpha = alpha
        self.lmbda = lmbda
        self.d_latent = d_latent

        self.embedder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 7, 2, 'same'),  # 100
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2), 2, padding='same'),  # 50
            ResidualBlock([32, 32, 64], [1, 3, 1], 2),
            tf.keras.layers.MaxPool2D((2, 2), 2, padding='same'),  # 25
            ResidualBlock([64, 64, 128], [1, 3, 1], 3),
            tf.keras.layers.MaxPool2D((2, 2), 2, padding='same',),  # 13
            ResidualBlock([128, 128, 256], [1, 3, 1], 4),
            tf.keras.layers.MaxPool2D((2, 2), 2, padding='same'),  # 7
            ResidualBlock([256, 256, 512], [1, 3, 1], 3),
            tf.keras.layers.AvgPool2D((7, 7), 7),  # 1
            tf.keras.layers.Reshape((512,)),
            tf.keras.layers.Dense(d_latent),
        ])

    def call(self, x, training=False):
        z = self.embedder(x, training=training)
        return tf.math.l2_normalize(z, axis=1)

    def call_on_identities(self, identities_x, training=False):
        n_identities = identities_x.shape[0]
        n_prints_per_identity = identities_x.shape[1]
        prints_x = tf.reshape(
            identities_x, [-1, INPUT_HEIGHT, INPUT_WIDTH, 1])
        prints_z = self.call(prints_x, training=training)
        identities_z = tf.reshape(
            prints_z, [n_identities, n_prints_per_identity, self.d_latent])
        return identities_z

    def triplet_loss(self, z_a, z_p, z_n):
        batch_sz = z_a.shape[0]
        positive_dist = tf.norm(z_a - z_p, axis=1)
        negative_dist = tf.norm(z_a - z_n, axis=1)
        J = positive_dist - negative_dist + self.alpha
        return tf.math.maximum(J, tf.zeros([batch_sz]))

    def softmax_loss(self, z_a, z_p):
        z_a_softmax = tf.nn.softmax(z_a, axis=1)
        z_p_softmax = tf.nn.softmax(z_p, axis=1)
        l = -tf.reduce_sum(z_a_softmax * tf.math.log(z_p_softmax), axis=1)
        return l

    def loss_function(self, z_a, z_p, z_n):
        l = self.triplet_loss(z_a, z_p, z_n) #+ self.lmbda * \
            #self.softmax_loss(z_a, z_p)
        return tf.reduce_sum(l)


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_sizes, repetitions):
        super(ResidualBlock, self).__init__()
        filters = filters * repetitions
        kernel_sizes = kernel_sizes * repetitions
        n_conv = len(filters)
        assert(n_conv == len(kernel_sizes))

        self.convolutions = tf.keras.Sequential()

        for i in range(n_conv):
            c = tf.keras.layers.Conv2D(
                filters[i], kernel_sizes[i], padding='same')
            b = tf.keras.layers.BatchNormalization()
            a = tf.keras.layers.ReLU()
            self.convolutions.add(c)
            self.convolutions.add(b)
            self.convolutions.add(a)

    def call(self, x, training=False):
        out = self.convolutions(x, training=training)
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, out.shape[3] - x.shape[3]]]) + out
