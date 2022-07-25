import os

# Reduce tensorflow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import Network
import LARS
import math
import DataGenerator_CIFAR10
import time

# Limit memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


class BYOL:
    """The Byol class performs training, forward and backward pass."""

    def __init__(self,
                 img_dims,
                 num_epochs,
                 batch_size,
                 training_data_size,
                 batches_per_epoch,
                 accumulation_steps,
                 tau_base,
                 opt,
                 use_GN_WS):

        self.img_dims = img_dims
        self.online_model = Network.build_model(self.img_dims, use_GN_WS, online=True)
        self.target_model = Network.build_model(self.img_dims, use_GN_WS, online=False)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.training_data_size = training_data_size
        self.batches_per_epoch = batches_per_epoch
        self.accumulation_steps = accumulation_steps
        self.tau_base = tau_base
        self.opt = opt
        if opt == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        elif opt == 'LARS':
            self.optimizer = LARS.LARSOptimizer(learning_rate=0.2,
                                                momentum=0.9,
                                                weight_decay=1 * 10 ** -6,
                                                eeta=0.001,  # The LARS coefficient is a hyperparameter
                                                epsilon=10 ** -6,
                                                use_nesterov=False,
                                                num_epochs=self.num_epochs,
                                                warm_up=10,
                                                batch_size=self.batch_size,
                                                name="LARSOptimizer")
        else:
            raise ValueError('Currently you have to select between Adam and LARS optimizers.')

    def loss_fn(self, on_pred_one, on_pred_two, tar_proj_one, tar_proj_two):
        """ Compute the Loss (Symmetric mean square error between normalized predictions and projections)"""
        norm_online_one = tf.math.l2_normalize(on_pred_one, axis=-1)
        norm_online_two = tf.math.l2_normalize(on_pred_two, axis=-1)
        norm_target_one = tf.math.l2_normalize(tar_proj_one, axis=-1)
        norm_target_two = tf.math.l2_normalize(tar_proj_two, axis=-1)

        loss1 = 2 - 2 * tf.math.reduce_sum(norm_online_one * norm_target_one, axis=-1)
        loss2 = 2 - 2 * tf.math.reduce_sum(norm_online_two * norm_target_two, axis=-1)
        return tf.math.reduce_mean(loss1 + loss2)

    def update_target_model(self, online_vars, target_vars, epoch):
        """ Update the target model based on the slow moving average of the online model"""
        tau = 1 - (1 - self.tau_base) * (tf.math.cos(math.pi * epoch / self.num_epochs) + 1) / 2  # Converges to 1
        online_vars = online_vars[:-8]  # Get relevant part of weights from the online model
        self.target_model.set_weights(tau * target_vars + (1 - tau) * online_vars)

    def batch_stats(self, idx, total_loss, t_batch_start, epoch_loss):
        """ Gives statistics about every batch as well as the progress of the epoch"""
        progress = round((idx + 1) * 100 / self.batches_per_epoch, 2)
        batch_loss = round(float(total_loss / self.accumulation_steps), 5)
        epoch_loss = round((epoch_loss * idx + float(total_loss / self.accumulation_steps)) / (idx + 1), 5)
        t_batch_time = round(time.time() - t_batch_start, 2)
        print(f"Progress: {progress}%  Batch loss: {batch_loss}   Batch time: {t_batch_time} sec", end='\r')
        return epoch_loss

    def epoch_stats(self, t_epoch_start, epoch_loss, epoch):
        """ Give some information about the current epoch"""
        t_epoch_end = time.time()
        t_epoch = round(t_epoch_end - t_epoch_start, 0)
        print(f"\nEpoch {epoch + 1} took {t_epoch} sec. Average loss: {epoch_loss}")

    def train(self, online_gen, target_gen):

        # Init both models with the same weights
        ''' This is done in the pytorch implementation but makes absolutely no sense '''
        # online_vars = np.array(self.online_model.get_weights(), dtype=object)
        # online_vars = online_vars[:-8]
        # self.target_model.set_weights(online_vars)

        for epoch in range(self.num_epochs):
            if self.opt == 'Lars':
                self.optimizer.update_current_epoch(epoch)  # Required for cosine decay in LARS Optimizer
            print(f"\nStart of epoch {epoch + 1}")
            t_epoch_start = time.time()
            epoch_loss = 0

            for idx in range(self.batches_per_epoch):

                # Set total loss to zero
                total_loss = 0
                t_batch_start = time.time()

                # Get trainable variables
                train_vars = self.online_model.trainable_variables

                # Create empty gradient list
                gradient_list = [tf.zeros_like(this_var) for this_var in train_vars]

                # Create empty summed gradient
                summed_gradients = []

                for batch_idx in range(self.accumulation_steps):
                    online_batch = online_gen[idx * self.accumulation_steps + batch_idx][0]
                    target_batch = target_gen[idx * self.accumulation_steps + batch_idx][0]

                    with tf.GradientTape() as tape:
                        # Get online network predictions
                        online_pred_one = self.online_model(online_batch, training=True)
                        online_pred_two = self.online_model(target_batch, training=True)

                        # Get target network projection (no gradient)
                        target_proj_one = tf.stop_gradient(self.target_model(target_batch, training=False))
                        target_proj_two = tf.stop_gradient(self.target_model(online_batch, training=False))

                        # Compute symmetrical loss
                        loss = self.loss_fn(online_pred_one, online_pred_two, target_proj_one, target_proj_two)

                    total_loss = total_loss + loss

                    # gradient of current batch (Clears the tape)
                    gradients = tape.gradient(loss, train_vars)

                    # Accumulate the gradient of the current mini-batch
                    summed_gradients = [gradient_old + gradient_new for gradient_old, gradient_new in
                                        zip(gradient_list, gradients)]

                # Update online network
                gradient_list = [this_grad for this_grad in summed_gradients]
                self.optimizer.apply_gradients(zip(gradient_list, train_vars))

                # Update target network
                self.update_target_model(np.array(self.online_model.get_weights(), dtype=object),
                                         np.array(self.target_model.get_weights(), dtype=object), epoch)

                # Print batch statistics
                epoch_loss = self.batch_stats(idx, total_loss, t_batch_start, epoch_loss)

            # Save the model of this epoch
            model_name = f'byol_train_epoch_{epoch}_loss_{epoch_loss}'
            tf.keras.models.save_model(self.online_model, f'saved_model/test/{model_name}.hdf5')

            # Print epoch statistics
            self.epoch_stats(t_epoch_start, epoch_loss, epoch)

            # Shuffle data at the end of epoch
            online_gen.on_epoch_end()
            target_gen.on_epoch_end()


def main():
    """ Main function calls the generator setup and calls the training procedure"""

    print("Create Generator...")
    mini_batch_size = 100
    online_gen, target_gen = DataGenerator_CIFAR10.generator_setup(batch_size=mini_batch_size)
    online_gen.on_epoch_end()
    target_gen.on_epoch_end()
    # DataGenerator_CIFAR10.visualize_batch()
    print("Generator Created!")

    print("Training model...")
    training = BYOL(img_dims=(32, 32, 3),
                    num_epochs=200,
                    batch_size=1000,
                    training_data_size=50000,
                    batches_per_epoch=50,
                    accumulation_steps=10,
                    tau_base=0.9995,
                    opt='LARS',
                    use_GN_WS=False)
    training.train(online_gen, target_gen)
    print("Model trained!")


if __name__ == '__main__':
    main()
