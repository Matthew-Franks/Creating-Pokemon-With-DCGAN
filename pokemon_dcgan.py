import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, losses, optimizers, Model


tf.random.set_seed(45)


class DCGANGenerator(Model):
    
    def __init__(self, latent_dim):
        
        super(DCGANGenerator, self).__init__()
        
        # First layer
        self.dense_1 = layers.Dense(units = latent_dim * 2 * 2, input_dim = latent_dim)
        self.activation_1 = layers.Activation('relu')
        self.reshape_1 = layers.Reshape(target_shape = (2, 2, latent_dim))
        
        # Second layer
        self.upsampling_2 = layers.UpSampling2D()
        self.conv_2 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_2 = layers.BatchNormalization(momentum = 0.7)
        self.activation_2 = layers.Activation('relu')

        # Third layer
        self.upsampling_3 = layers.UpSampling2D()
        self.conv_3 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_3 = layers.BatchNormalization(momentum = 0.7)
        self.activation_3 = layers.Activation('relu')

        # Fourth layer
        self.upsampling_4 = layers.UpSampling2D()
        self.conv_4 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_4 = layers.BatchNormalization(momentum = 0.7)
        self.activation_4 = layers.Activation('relu')

        # Fifth layer
        self.upsampling_5 = layers.UpSampling2D()
        self.conv_5 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_5 = layers.BatchNormalization(momentum = 0.7)
        self.activation_5 = layers.Activation('relu')

        # Sixth layer
        self.upsampling_6 = layers.UpSampling2D()
        self.conv_6 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_6 = layers.BatchNormalization(momentum = 0.8)
        self.activation_6 = layers.Activation('relu')

        # Seventh layer
        self.upsampling_7 = layers.UpSampling2D()
        self.conv_7 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_7 = layers.BatchNormalization(momentum = 0.7)
        self.activation_7 = layers.Activation('relu')

        # Eighth layer
        self.conv_8 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_8 = layers.BatchNormalization(momentum = 0.7)
        self.activation_8 = layers.Activation('relu')

        # Ninth layer
        self.conv_9 = layers.Conv2D(filters = 3, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.activation_9 = layers.Activation('tanh')
    
    
    def call(self, z):
        
        '''
        This function will be called by the class to invoke the layers
        defined in the __init__() function.
        '''

        z = self.dense_1(z)
        z = self.activation_1(z)
        z = self.reshape_1(z)

        z = self.upsampling_2(z)
        z = self.conv_2(z)
        z = self.norm_2(z)
        z = self.activation_2(z)

        z = self.upsampling_3(z)
        z = self.conv_3(z)
        z = self.norm_3(z)
        z = self.activation_3(z)

        z = self.upsampling_4(z)
        z = self.conv_4(z)
        z = self.norm_4(z)
        z = self.activation_4(z)

        z = self.upsampling_5(z)
        z = self.conv_5(z)
        z = self.norm_5(z)
        z = self.activation_5(z)

        z = self.upsampling_6(z)
        z = self.conv_6(z)
        z = self.norm_6(z)
        z = self.activation_6(z)

        z = self.upsampling_7(z)
        z = self.conv_7(z)
        z = self.norm_7(z)
        z = self.activation_7(z)

        z = self.conv_8(z)
        z = self.norm_8(z)
        z = self.activation_8(z)

        z = self.conv_9(z)
        
        output = self.activation_9(z)

        return output


class DCGANDiscriminator(Model):
    
    def __init__(self, image_shape):
        
        super(DCGANDiscriminator, self).__init__()

        # First layer
        self.conv_1 = layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = 2, padding = 'same')
        self.activation_1 = layers.LeakyReLU(alpha = 0.2)
        self.dropout_1 = layers.Dropout(rate = 0.25)

        # Second layer
        self.conv_2 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 2, padding = 'same')
        self.zeropad_2 = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))
        self.norm_2 = layers.BatchNormalization(momentum = 0.8)
        self.activation_2 = layers.LeakyReLU(alpha = 0.2)
        self.dropout_2 = layers.Dropout(rate = 0.25)

        # Third layer
        self.conv_3 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = 2, padding = 'same')
        self.norm_3 = layers.BatchNormalization(momentum = 0.8)
        self.activation_3 = layers.LeakyReLU(alpha = 0.1)
        self.dropout_3 = layers.Dropout(rate = 0.25)

        # Fourth layer
        self.conv_4 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = 1, padding = 'same')
        self.norm_4 = layers.BatchNormalization(momentum = 0.8)
        self.activation_4 = layers.LeakyReLU(alpha = 0.1)
        self.dropout_4 = layers.Dropout(rate = 0.25)

        # Fifth layer
        self.flatten = layers.Flatten()
        self.dense_5 = layers.Dense(units = 1)
        self.activation_5 = layers.Activation('sigmoid')
    
    
    def call(self, z):
        
        '''
        This function will be called by the class to invoke the layers
        defined in the __init__() function.
        '''
        
        z = self.conv_1(z)
        z = self.activation_1(z)
        z = self.dropout_1(z)

        z = self.conv_2(z)
        z = self.zeropad_2(z)
        z = self.norm_2(z)
        z = self.activation_2(z)
        z = self.dropout_2(z)

        z = self.conv_3(z)
        z = self.norm_3(z)
        z = self.activation_3(z)
        z = self.dropout_3(z)

        z = self.conv_4(z)
        z = self.norm_4(z)
        z = self.activation_4(z)
        z = self.dropout_4(z)

        z = self.flatten(z)

        z = self.dense_5(z)
        
        output = self.activation_5(z)

        return output


def train_dcgan(generative_model, discriminator_model, loss_, generative_optimizer, discriminator_optimizer, batch_size, latent_dim_, batch):
    
    '''
    Define the latent vector and instantiate the real and fake labels.
    Generate output using the generator. Test and compare these results
    with the discriminator against the real images. Calculate the loss
    from the two outputs.
    '''
    
    latent_vector = tf.random.normal(shape = (batch_size, latent_dim_))
    fake_labels = tf.zeros(shape = (batch_size, 1))
    valid_labels = tf.ones(shape = (batch_size, 1))
    
    with tf.GradientTape() as tape:
        
        generative_output = generative_model(latent_vector, training = True)
        fake_output = discriminator_model(generative_output, training = True)
        real__output = discriminator_model(batch, training = True)
        fake_loss = loss_(fake_labels, fake_output)
        real_loss = loss_(valid_labels, real__output)
        discriminator_loss = tf.multiply(tf.add(real_loss, fake_loss), 1.0)
        
    trainable_variables = generative_model.trainable_variables + discriminator_model.trainable_variables
    gradients = tape.gradient(discriminator_loss, trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, trainable_variables))

    with tf.GradientTape() as tape:
        
        generative_output = generative_model(latent_vector, training = True)
        fake_output = discriminator_model(generative_output, training = True)
        generative_loss = loss_(valid_labels, fake_output)
        
    trainable_variables = generative_model.trainable_variables
    gradients = tape.gradient(generative_loss, trainable_variables)
    generative_optimizer.apply_gradients(zip(gradients, trainable_variables))

    return generative_loss, discriminator_loss


def save_images(epoch, latent_dim_, generative_model):
    
    '''
    This function is responsible for producing and saving the images created
    by the generator to a local file for analysis.
    '''
    
    rows, columns = 5, 5
    latent_vector = tf.random.normal(shape = (rows * columns, latent_dim_))
    
    generated_images = generative_model(latent_vector, training = True)
    generated_images = generated_images * 0.5 + 0.5
    
    fig, axis = plt.subplots(rows, columns, figsize = (15, 15))
    count = 0
    
    for i in range(rows):
        
        for j in range(columns):
            
            axis[i, j].imshow(generated_images[count])
            axis[i, j].axis('off')
            count += 1
            
    fig.savefig('data/output/{0}.png'.format(epoch))
    
    plt.close()


def main():

    # Creation of variables and pathways
    data = []
    image_shape_ = (128, 128, 3)
    latent_dim_ = 128
    batch_size = 16
    data_directory = 'data/input/'
    checkpoint_directory = 'data/checkpoint/'
    
    # Resizing the images for analysis and putting them into 'data' variable
    for filename in os.listdir(path = data_directory):
        
        image_path = os.path.join(data_directory, filename)
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.shape[0] <= 120 or image.shape[1] <= 120:
            
            image = np.pad(image, pad_width = ((4, 4), (4, 4), (0, 0)), mode = 'constant')
            
        else:
            
            image = cv2.resize(image, (128, 128))
            
        data.append(image)
        
    data = np.array(data)
    
    # This is where we create the training batch
    training_batch = tf.data.Dataset.from_tensor_slices(data).shuffle(1000).batch(batch_size, drop_remainder = True)
    
    # Initialize our two models
    generative_model = DCGANGenerator(latent_dim = latent_dim_)
    discriminator_model = DCGANDiscriminator(image_shape = image_shape_)
    
    # Define our loss function and instnatiate our optimizers
    loss_ = losses.BinaryCrossentropy()
    generative_optimizer = optimizers.Adam(learning_rate = 0.001)
    discriminator_optimizer = optimizers.Adam(learning_rate = 0.0001)
    
    # Create our checkpoint system so we don't lose our progress
    checkpoint_ = tf.train.Checkpoint(step = tf.Variable(0), optimizer_g = generative_optimizer, optimizer_d = discriminator_optimizer, g_model = generative_model, d_model = discriminator_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint = checkpoint_, directory = checkpoint_directory, max_to_keep = 1)
    checkpoint_.restore(checkpoint_manager.latest_checkpoint)
    
    # Restore our progress
    if checkpoint_manager.latest_checkpoint:
        
        print('Restored from last checkpoint : {0}'.format(int(checkpoint_.step)))
        
    checkpoint_epoch = int(checkpoint_.step)
    epochs = 10000
    
    # To keep track of the time between epochs
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    # This is the actual training of our models where we aim to minimize our
    # loss by calling the train_dcgan() function
    for epoch in range(checkpoint_epoch, epochs):
        
        generative_loss = metrics.Mean()
        discriminator_loss = metrics.Mean()
        
        for batch in training_batch:
            
            batch = (tf.cast(batch, dtype = 'float32') - 127.5) / 127.5
            generative_loss_, discriminator_loss_ = train_dcgan(generative_model, discriminator_model, loss_, generative_optimizer, discriminator_optimizer, batch_size, latent_dim_, batch)
            generative_loss.update_state(generative_loss_)
            discriminator_loss.update_state(discriminator_loss_)
            
        checkpoint_.step.assign_add(1)
        checkpoint_manager.save()
        
        print('\nEpoch: {0}, Loss (G): {1}, Loss (D): {2}'.format(epoch, generative_loss.result(), discriminator_loss.result()))
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        
        if epoch % 2 == 0:
            
            save_images(epoch, latent_dim_, generative_model)


# When the file is ran, it will call the main function first
if __name__ == '__main__':
    
    main()