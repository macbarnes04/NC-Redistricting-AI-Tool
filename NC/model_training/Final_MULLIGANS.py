import json
import tensorflow as tf
import wandb
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
from matplotlib import pyplot as plt
import os

wandb.init(project="MULLIGANS")

# opening JSON File
f = open("RI1000testrun2.json")

# Returns JSON object as a dictionary
raw_data = json.load(f)

# --------------------------------------------------------------------------------------------
# LOADING DATA INTO TENSORS:
#
#
# PLANS:
equivalency_files = raw_data["EquivalencyFiles"]
equivalency_files = tf.convert_to_tensor(equivalency_files) - 1
# minus 1 since there are districts 1-13 (now 0-12)
print(equivalency_files)
#
# SCORES:
#
# Compactness
compactness_files = raw_data["CompactnessScores"]
compactness_files = tf.convert_to_tensor(compactness_files)
#
# Mean Median
mean_median_files = raw_data["PartisanBiasArray"]
mean_median_files = tf.convert_to_tensor(mean_median_files)
#
# Stacking / Scaling Metrics
#
scores = tf.stack([0.01*tf.cast(compactness_files, tf.float32), mean_median_files], axis=-1)
#
# --------------------------------------------------------------------------------------------
# DEFINING TRAINING AND TESTING DATA
#
train_size = int(len(equivalency_files)*0.8)
test_size = int(len(equivalency_files) - train_size)
#
# colon denotes which side of split to look at (this is called slicing)
x_train = equivalency_files[:train_size]
x_test = equivalency_files[train_size:]
#
y_train = scores[:train_size]
y_test = scores[train_size:]
#
# --------------------------------------------------------------------------------------------
# ONE-HOT ENCODE (CATEGORALIZE) X_TRAIN/X_TEST DATA
#
x_train = tf.keras.utils.to_categorical(x_train, num_classes=13)
x_test = tf.keras.utils.to_categorical(x_test, num_classes=13)
# no need to do this with the y data since it is a float
#
# --------------------------------------------------------------------------------------------
# DATAPOINT AE:
#
# Taking 13 Dimensional Input, Scaling Down to 4 Dimensions
# Bypasses Need For _____
embed_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(None, 13,), activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='tanh'),
])
#
# Rescales Data After Going Through Main CVAE Network
embed_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(None, 4), activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(13, activation='softmax'),
])
#
#tensorboard1 = TensorBoard(log_dir="logs1/{}".format(time()))
#
embed_ae = tf.keras.models.Sequential([embed_encoder, embed_decoder])
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#
embed_ae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
embed_ae.fit(x_train, x_train, epochs=5, batch_size=4)
embed_ae.evaluate(x_test, x_test, verbose=2)
embed_ae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
#
embed_encoder.trainable = False
embed_decoder.trainable = False
#
# --------------------------------------------------------------------------------------------
# MAIN ENCODER MODEL
#
encoder_input = tf.keras.layers.Input(shape=(2706, 13))
embedded_encoder_input = embed_encoder(encoder_input)
flattened_encoder_embedding = tf.keras.layers.Flatten()(embedded_encoder_input)
encoder_dense_1 = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(flattened_encoder_embedding)
encoder_dense_1_1 = tf.keras.layers.Dense(1536, activation=tf.keras.activations.swish)(encoder_dense_1)
encoder_dense_2 = tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish)(encoder_dense_1_1)
encoder_dense_2_2 = tf.keras.layers.Dense(512, activation=tf.keras.activations.swish)(encoder_dense_2)
encoder_dense_3 = tf.keras.layers.Dense(256, activation=tf.keras.activations.swish)(encoder_dense_2_2)
#
mean = tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish)(encoder_dense_3)
logvar = tf.keras.layers.Dense(1024)(encoder_dense_3)
# the 1024 is an arbitrary variable
metrics = tf.keras.layers.Dense(2, activation=tf.keras.activations.swish)(encoder_dense_3)
#
outputs = [
    mean,
    logvar,
    metrics
]
#
encoder = tf.keras.models.Model(inputs=encoder_input, outputs=outputs)
#
# --------------------------------------------------------------------------------------------
# MAIN DECODER MODEL
#
decoder_inputs = [
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Input(shape=(2,))
]
decoder_input = tf.concat(decoder_inputs, -1)
decoder_dense_1 = tf.keras.layers.Dense(256, activation=tf.keras.activations.swish)(decoder_input)
decoder_dense_1_1 = tf.keras.layers.Dense(512, activation=tf.keras.activations.swish)(decoder_dense_1)
decoder_dense_2 = tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish)(decoder_dense_1_1)
decoder_dense_2_2 = tf.keras.layers.Dense(1536, activation=tf.keras.activations.swish)(decoder_dense_2)
decoder_dense_3 = tf.keras.layers.Dense(2048, activation=tf.keras.activations.swish)(decoder_dense_2_2)
decoder_dense_4 = tf.keras.layers.Dense(2706 * 4, activation=tf.keras.activations.relu)(decoder_dense_3)
reshape_decoder = tf.keras.layers.Reshape((2706, 4))(decoder_dense_4)
unembed_decoder = embed_decoder(reshape_decoder)
#
decoder = tf.keras.models.Model(inputs=decoder_inputs, outputs=unembed_decoder)
#
encoder.summary()
decoder.summary()
#
# --------------------------------------------------------------------------------------------
# DEFINING TOTAL AUTOENCODER
#
autoencoder_input = tf.keras.layers.Input(shape=(2706, 13))
_mean, _logvar, metrics = encoder(autoencoder_input)
reconstructed = decoder([_mean * tf.random.normal(tf.shape(_mean)) + tf.exp(_logvar * 0.5), metrics])
outputs = [reconstructed, _mean, _logvar, metrics]
#
autoencoder = tf.keras.models.Model(inputs=autoencoder_input, outputs=outputs)
#
#
# --------------------------------------------------------------------------------------------
# DEFINING LOSS FUNCTION
#
# y_true will be a list: [input, expected_metrics]
# y_pred will be a list: [reconstructed, mean, logvar, metrics]
# We want latent to resemble a sampling from N(0, 1) and metrics to match the expected metrics
@tf.function
def loss_fn(y_true, y_pred):
    input, expected_metrics = y_true[0], y_true[1]
    reconstructed, mean, logvar, metrics = y_pred[0], y_pred[1], y_pred[2], y_pred[3]

    return (tf.reduce_mean((input - reconstructed)**2),
           tf.reduce_mean((metrics - expected_metrics)**2),
           tf.reduce_mean(0.5*(1 - (0.5*logvar)**2 + tf.exp(0.5*logvar)**2 + mean**2)))
#
# --------------------------------------------------------------------------------------------
# Custom Training Loop
# from google.colab.output import clear




tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)

batch_size = 25

optimizer = tf.keras.optimizers.Adamax()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

train_accuracy_results = []
train_loss_results = []
rloss_results = []
mloss_results = []
kldivloss_results = []

epochs= 15

for epoch in range(epochs):
    # clear()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_accuracy_1 = tf.keras.metrics.Mean()
    epoch_loss = tf.keras.metrics.Mean()
    wandb.config.epochs = 15
    wandb.config.batch_size = 25
    print(f"Start Epoch {epoch}")
    for step, (images, metrics) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = autoencoder(images)
            rloss, mloss, kldivloss = loss_fn([images, metrics], predictions)
            loss = rloss + mloss + kldivloss
        grads = tape.gradient(loss, autoencoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
        epoch_loss.update_state(loss)
        epoch_accuracy.update_state(tf.math.argmax(images, axis=-1), predictions[0])
        wandb.log({'accuracy': epoch_accuracy.result().numpy(), 'loss': epoch_loss.result().numpy()})
    train_loss_results.append(epoch_loss.result())
    train_accuracy_results.append(epoch_accuracy.result())
    rloss_results.append(rloss)
    mloss_results.append(mloss)
    kldivloss_results.append(kldivloss)
    print(epoch_accuracy.result())
    print(epoch_loss.result())
    print(mloss)
    print(rloss)
    print(kldivloss)


        # if step > -1:
        #     print (f"Training loss for batch {step:04d}/{len(train_dataset):04d}: {float(loss)}; Recon loss: {float(rloss)}; Metric loss: {float(mloss)}; KLD loss: {float(kldivloss)}")
        #     fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        #     fig.suptitle("Training Metrics")
        #
        #     axes[0].set_ylabel("Loss", fontsize=14)
        #     axes[0].plot(loss)
        #
        #     axes[1].set_ylabel("Accuracy", fontsize=14)
        #     axes[1].set_xlabel("Epoch", fontsize=14)
        #     axes[1].plot(train_accuracy_results)
        #     plt.show()

    # optimizer.minimize(loss, autoencoder.trainable_variables, tape=tape)
#
# --------------------------------------------------------------------------------------------


plt.plot(range(epochs), train_loss_results, 'tab:purple', label='Training Loss')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.savefig('cvae_loss.png')

plt.clf()

plt.plot(range(epochs), train_accuracy_results, 'tab:purple', label='Training Accuracy')

plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.savefig('cvae_accuracy.png')

plt.clf()
# --------------------------------------------------------------------------------------------
# SAVING MODEL
decoder.save("RIDecoder.h5")