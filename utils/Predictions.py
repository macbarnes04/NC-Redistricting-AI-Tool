import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
import numpy as np

MM = .0285
CE = 500

def prediction(MM_value, CE_value):
    metrics_matrix = [MM_value, CE_value]
    metrics = tf.convert_to_tensor(metrics_matrix)
    metrics = tf.reshape(
        metrics, shape=(1, 2), name=None
    )
    print(metrics)

    latent_tensor = tf.random.normal(shape=(1024,), mean=0, stddev=2, dtype=tf.float32, seed=None, name=None)
    latent_tensor = tf.reshape(
        latent_tensor, shape=(1, 1024), name=None
    )
    print(latent_tensor)

    decoder_inputs = [
        latent_tensor,
        metrics
    ]

    ML_model = tf.keras.models.load_model('mulligans_decoder.h5')
    output = ML_model.predict(decoder_inputs)
    n = np.squeeze(np.argmax(output, axis=-1).tolist())
    # i = range(1, len(n)+1)
    # pd.concat(i, n, headers=[])
    # output = [f.write(f"{i + 1}, {n + 1}" + "\n") for i, n in enumerate(np.argmax(output, axis=-1)[0].tolist())]
    return(n)
output = prediction(MM, CE)
print(output)
print(len(output))


