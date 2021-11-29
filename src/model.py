import tensorflow as tf

class FingNet(tf.keras.Model):
    def __init__(self, alpha, lmbda, d_latent):
        super(FingNet, self).__init__()
        self.alpha = alpha
        self.lmbda = lmbda

        self.embedder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 7, 2, 'same'), # 100
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2,2), 2, padding='same'), # 50
            ResidualBlock([32, 32, 64], [1, 3, 1], 2),
            tf.keras.layers.MaxPool2D((2,2), 2, padding='same'), # 25
            ResidualBlock([64, 64, 128], [1, 3, 1], 3),
            tf.keras.layers.MaxPool2D((2,2), 2, padding='same',), # 13
            ResidualBlock([128, 128, 256], [1, 3, 1], 4),
            tf.keras.layers.MaxPool2D((2,2), 2, padding='same'), # 7
            ResidualBlock([256, 256, 512] , [1, 3, 1], 3),
            tf.keras.layers.AvgPool2D((7,7), 7), # 1
            tf.keras.layers.Reshape((512,)),
            tf.keras.layers.Dense(d_latent),
        ])

    def call(self, x):
        z = self.embedder(x)
        return tf.math.l2_normalize(z, axis = 1)

    def triplet_loss(self, z_a, z_p, z_n):
        batch_sz = z_a.shape[0]
        positive_dist = tf.norm(z_a - z_p)
        negative_dist = tf.norm(z_a - z_n)
        J = positive_dist - negative_dist + self.alpha
        return tf.math.maximum(J, tf.zeros([batch_sz]))

    def softmax_loss(self, z_a, z_p, z_n):
        l = tf.losses.categorical_crossentropy(z_a, z_p)
        return l
    
    def loss_function(self, z_a, z_p, z_n):
        l = self.triplet_loss(z_a, z_p, z_n) #+ self.lmbda * self.softmax_loss(z_a, z_p, z_n)
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
            c = tf.keras.layers.Conv2D(filters[i], kernel_sizes[i], padding='same')
            b = tf.keras.layers.BatchNormalization()
            a = tf.keras.layers.ReLU()
            self.convolutions.add(c)
            self.convolutions.add(b)
            self.convolutions.add(a)
        
        
    def call(self, x):
        out = self.convolutions(x)
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, out.shape[3] - x.shape[3]]]) + out            
