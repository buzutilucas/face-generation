import tensorflow as tf

class DCGAN:
    """
    Deep Convolutional Generative Adversarial Networks or DCGANs are models
    that can generate new datasets.
    """
    def __init__(self, dataset_shape, z_dim, beta1=.0):
        """
        Create the model.
        :param dataset_shape: A tuple of 4-D (batch, width, height, channels).
        :param z_dim: The dimension of Z.
        :param ckpt: Checkpoint path.
        :param beta1: The exponential decay rate for the 1st moment in the optimizer.
        :param is_train: Condition for training (True or False)
        """
        self.FLAG_d = 'discriminator'

        with tf.Graph().as_default() as self.graph:
            _, image_width, image_height, image_channels = dataset_shape
            self.input_real, self.input_z, self.lr = self.model_inputs(image_width,
                                                                       image_height,
                                                                       image_channels,
                                                                       z_dim)
            g_d = self.model_loss(self.input_real, self.input_z, image_channels)
            self.d_loss, self.g_loss, self.d_logits_real, self.d_logits_fake = g_d
            self.d_train_opt, self.g_train_opt = self.model_opt(self.d_loss,
                                                                self.g_loss,
                                                                self.lr,
                                                                beta1)

    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs.
        :param image_width: The input image width.
        :param image_height: The input image height.
        :param image_channels: The number of image channels.
        :param z_dim: The dimension of Z.
        :return: Tuple of (tensor of real input images, tensor of z data, learning rate).
        """
        input_ = tf.placeholder(tf.float32,
                                shape=[None, image_width, image_height, image_channels],
                                name='input')
        input_z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z_input')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        return input_, input_z, learning_rate

    def discriminator(self, images, reuse=False):
        """
        Create the discriminator network.
        :param images: Tensor of input image(s).
        :param reuse: Boolean if the weights should be reused.
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator).
        """
        alpha = 0.2
        with tf.variable_scope('discriminator', reuse=reuse):
          # Input is 28x28x?
          # 1st Convolutional layer
          conv1 = tf.layers.conv2d(images, filters=32, kernel_size=(5,5),
                               strides=(2,2), padding='same')
          conv1 = tf.nn.leaky_relu(conv1, alpha=alpha) # or tf.maximum(alpha*x, x)
          # 14x14x32 now

          # 2nd Convolutional layer
          conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(5,5),
                                   strides=(2,2), padding='same')
          conv2 = tf.layers.batch_normalization(conv2, training=True) # Batch normalizing
          conv2 = tf.nn.leaky_relu(conv2, alpha=alpha) # or tf.maximum(alpha*x, x)
          # 7x7x64 now

          # Flatten Layer
          flat = tf.reshape(conv2, (-1, 7*7*64))
          logits = tf.layers.dense(flat, 1)
          out = tf.nn.sigmoid(logits)

        return out, logits

    def generator(self, z, out_channel_dim, is_train=True):
        """
        Create the generator network.
        :param z: Input z.
        :param out_channel_dim: The number of channels in the output image.
        :param is_train: Boolean if generator is being used for training.
        :return: The tensor output of the generator.
        """
        alpha = 0.2
        with tf.variable_scope('generator', reuse=not is_train):
          # z Layers
          x = tf.layers.dense(z, 7*7*64)
          x = tf.reshape(x, (-1, 7, 7, 64))
          x = tf.layers.batch_normalization(x, training=is_train)
          x = tf.nn.leaky_relu(x, alpha=alpha)
          # 7x7x64 now

          # 1st Transpose Convolutional
          conv1_t = tf.layers.conv2d_transpose(x, filters=32, kernel_size=(5,5),
                                               strides=(2,2), padding='same')
          conv1_t = tf.layers.batch_normalization(conv1_t, training=is_train)
          conv1_t = tf.nn.leaky_relu(conv1_t, alpha=alpha)
          # 14x14x32 now

          # 2nd Transpose Convolutional
          logits = tf.identity(
              tf.layers.conv2d_transpose(conv1_t, filters=out_channel_dim,
                                         kernel_size=(5,5), strides=(2,2),
                                         padding='same'), name='conv2_t')
          out = tf.nn.tanh(logits)
          # 28x28x? now

        return out

    def model_loss(self, input_real, input_z, out_channel_dim):
        """
        Get the loss for the discriminator and generator.
        :param input_real: Images from the real dataset.
        :param input_z: Z input.
        :param out_channel_dim: The number of channels in the output image.
        :return: A tuple of (discriminator loss, generator loss,
                 discriminator logits real, discriminator logits fake).
        """
        g_model = self.generator(input_z, out_channel_dim)
        d_model_real, d_logits_real = self.discriminator(input_real)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                    labels=tf.ones_like(d_model_real)*0.9))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.zeros_like(d_model_fake)))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=tf.ones_like(d_model_fake)))

        return d_loss, g_loss, d_logits_real, d_logits_fake

    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations.
        :param d_loss: Discriminator loss Tensor.
        :param g_loss: Generator loss Tensor.
        :param learning_rate: Learning Rate Placeholder.
        :param beta1: The exponential decay rate for the 1st moment in the optimizer.
        :return: A tuple of (discriminator training operation, generator training operation).
        """
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

        # Optimize
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
          d_opt = tf.train.AdamOptimizer(
              learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
          g_opt = tf.train.AdamOptimizer(
              learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_opt, g_opt
