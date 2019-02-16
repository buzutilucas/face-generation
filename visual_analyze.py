import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Filter_Analyses():
    """
    Analyzing how CNN sees.
    """
    def __init__(self, model=None, sess=None):
        self.model = model
        self.sess = sess
        self.FALG_Leaky_ReLU = True

    def list_conv_layer_names(self):
        """
        Create a list of names for the operations in the graph,
        where the operator-types are 'Conv2D'.
        """
        return [op.name for op in self.model.graph.get_operations() if op.type=='Conv2D']

    def optimize_image(self, conv_id=None, feature=0, num_iterations=30, show_progress=True):
        """
        Find an image that maximizes the feature
        given by the conv_id and feature number.
        """
        # Create the loss function that must be maximized
        if conv_id is None and self.model.FLAG_d == 'discriminator':
            loss = tf.reduce_mean(self.model.d_logits_real)
        else:
            # Get the name of the convolutional operator.
            layer = self.list_conv_layer_names()
            conv_name = layer[conv_id]

            # Get a reference to the tensor that is output by the
            # operator. Note that ":0" is added to the name for this.
            tensor = self.model.graph.get_tensor_by_name(conv_name + ":0")

            with self.model.graph.as_default():
                # The loss-function is the average of all the
                # tensor-values for the given feature. This
                # ensures that we generate the whole input image.
                # You can try and modify this so it only uses
                # a part of the tensor.
                if self.FALG_Leaky_ReLU:
                    tensor = tf.nn.leaky_relu(tensor, 0.2)
                loss = tf.reduce_mean(tensor[:,:,:,feature])


        # Get the gradient for the loss-function with regard to
        # the input image. This creates a mathematical
        # function for calculating the gradient.
        gradient = tf.gradients(loss, self.model.input_real)

        # Generate a random image of the same size as the raw input
        image = np.random.uniform(size=self.model.input_real.get_shape().as_list()[1:]) + 128
        image = np.expand_dims(image, axis=0)


        # Perform a number of optimization iterations to find
        # the image that maximizes the loss-function.
        for i in range(num_iterations):
            feed = {self.model.input_real: image}

            # Calculate the gradient and the loss values
            grad, loss_value = self.sess.run([gradient, loss], feed_dict=feed)

            # Squeeze the dimensionality for the gradient-array.
            grad = grad[0]

            # Calculate the step-size for updating the image.
            # This step-size was found to give fast convergence.
            # The addition of 1e-8 is to protect from div-by-zero.
            step_size = 1.0 / (grad.std() + 1e-8)

            # Update the image by adding the scaled gradient
            # This is called gradient ascent.
            image += step_size * grad

            # Ensure all pixel-values in the image are between 0 and 255.
            image = np.clip(image, 0.0, 255.0)

            if show_progress:
                print("Iteration:", i)
                # Print statistics for the gradient.
                print("Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}".format(grad.min(), grad.max(), step_size))
                # Print the loss-value.
                print("Loss:", loss_value)
                # Newline.
                print()
        return image.squeeze()

    def normalize_image(self, x):
        """
        This function normalizes an image so its pixel-values are between 0.0 and 1.0.
        """
        # Get the min and max values for all pixels in the input.
        x_min = x.min()
        x_max = x.max()

        # Normalize so all values are between 0.0 and 1.0
        x_norm = (x - x_min) / (x_max - x_min)

        return x_norm

    def plot_image(self, image):
        """
        This function plots a single image.
        """
        # Normalize the image so pixels are between 0.0 and 1.0
        img_norm = self.normalize_image(image)

        # Plot the image.
        plt.imshow(img_norm, interpolation='nearest')
        plt.show()

    def plot_images10(self, images, smooth=True):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Create figure with sub-plots.
        fig, axes = plt.subplots(2, 3)

        # Adjust vertical spacing.
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        # For each entry in the grid.
        for i, ax in enumerate(axes.flat):
            # Get the i'th image and only use the desired pixels.
            img = images[i, :, :]

            # Plot the image.
            ax.imshow(img, interpolation=interpolation, cmap='binary')

            # Remove ticks.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_images(self, images, show_size=100, smooth=True):
        """
        The show_size is the number of pixels to show for each image.
        The max value is 299.
        """

        # Create figure with sub-plots.
        fig, axes = plt.subplots(2, 3)

        # Adjust vertical spacing.
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        # Interpolation type.
        if smooth: # Use interpolation to smooth pixels
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # For each entry in the grid.
        for i, ax in enumerate(axes.flat):
            # Get the i'th image and only use the desired pixels.
            img = images[i, 0:show_size, 0:show_size, :]

            # Normalize the image so its pixels are between 0.0 and 1.0
            img_norm = self.normalize_image(img)

            # Plot the image.
            ax.imshow(img_norm, interpolation=interpolation)

            # Remove ticks.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def optimize_images(self, conv_id=None, num_iterations=30, show_size=100):
        """
        Find 6 images that maximize the 6 first features in the layer
        given by the conv_id.
        """

        # Which layer are we using?
        if conv_id is None:
            print("Final fully-connected layer before sigmoid.")
        else:
            print("Layer:", self.list_conv_layer_names()[conv_id])

        # Initialize the array of images.
        images = []

        # For each feature do the following. Note that the
        # last fully-connected layer only supports numbers
        # between 1 and 1000, while the convolutional layers
        # support numbers between 0 and some other number.
        # So we just use the numbers between 1 and 7.
        for feature in range(1,7):
            print("Optimizing image for feature no.", feature)

            # Find the image that maximizes the given feature
            # for the network layer identified by conv_id (or None).
            image = self.optimize_image(conv_id=conv_id, feature=feature,
                                        show_progress=False,
                                        num_iterations=num_iterations)

            # Squeeze the dim of the array.
            image = image.squeeze()

            # Append to the list of images.
            images.append(image)

        # Convert to numpy-array so we can index all dimensions easily.
        return np.array(images)
