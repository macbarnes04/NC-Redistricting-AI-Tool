import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
DATA_PATH = "NC/data/processed_v2_data_100.npz" # Make sure this points to your new file!
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

print(f"Loading data from {DATA_PATH}...")
with np.load(DATA_PATH) as data:
    # Load and process Maps (X)
    raw_maps = data['maps'] # Shape: (N, 2706)
    # One-hot encode maps: (N, 2706) -> (N, 2706, 13)
    x_train_all = tf.one_hot(raw_maps, depth=13, axis=-1, dtype=tf.float32)
    
    # Load and process Features (Y) - The 6 metrics
    y_train_all = data['features'].astype('float32') # Shape: (N, 6)
    
    # Load Adjacency Matrix (A)
    # We need to normalize it for the GCN: A_norm = D^-0.5 * (A + I) * D^-0.5
    adj = data['adjacency'].astype('float32')
    # Add self-loops
    adj = adj + np.eye(adj.shape[0], dtype='float32')
    # Calculate degree matrix D
    d = np.diag(np.sum(adj, axis=1))
    # Inverse square root of D
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # A_hat = D^-0.5 * A * D^-0.5
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    # Convert to Tensor (Constant for all batches)
    ADJ_TENSOR = tf.constant(adj_norm, dtype=tf.float32)
    
    # Raw Adjacency for Loss Function (Binary, not normalized)
    ADJ_RAW = tf.constant(data['adjacency'], dtype=tf.float32)

# Split Data (80/20)
split_idx = int(len(x_train_all) * 0.8)
x_train, x_test = x_train_all[:split_idx], x_train_all[split_idx:]
y_train, y_test = y_train_all[:split_idx], y_train_all[split_idx:]

print(f"Data Loaded. Train Shape: {x_train.shape}, Test Shape: {x_test.shape}")

# Create Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# 2. DEFINE CUSTOM LAYERS (GCN)
# ---------------------------------------------------------
class GraphConv(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(GraphConv, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Weights: (Input_Feat_Dim, Output_Units)
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        # inputs is [Node_Features] (Batch, Nodes, Feats)
        # 1. Feature Transformation: H * W
        x = tf.matmul(inputs, self.w) 
        # 2. Feature Propagation: A_norm * (H * W)
        # We use sparse matrix multiplication logic or broadcasting
        # Since A is (Nodes, Nodes) and x is (Batch, Nodes, Units)
        out = tf.einsum('nm,bmu->bnu', ADJ_TENSOR, x)
        
        if self.activation:
            out = self.activation(out)
        return out

# 3. BUILD THE MODEL (GNN-CVAE)
# ---------------------------------------------------------
LATENT_DIM = 64 # Size of the "bottleneck" code

# --- ENCODER (GNN) ---
enc_input = tf.keras.Input(shape=(2706, 13))

# GNN Layers "see" the map structure
h = GraphConv(32, activation='relu')(enc_input)
h = GraphConv(16, activation='relu')(h)
h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(512, activation='relu')(h)

# Latent Space (Mean & LogVar)
z_mean = tf.keras.layers.Dense(LATENT_DIM)(h)
z_log_var = tf.keras.layers.Dense(LATENT_DIM)(h)

# Sampling Function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

# Auxiliary Output: Predict the Metrics from the Map (Verification)
metric_pred = tf.keras.layers.Dense(6)(h)

encoder = tf.keras.Model(enc_input, [z_mean, z_log_var, z, metric_pred], name="encoder")

# --- DECODER (MLP) ---
# Inputs: Latent Vector + Target Metrics (Condition)
lat_input = tf.keras.Input(shape=(LATENT_DIM,))
cond_input = tf.keras.Input(shape=(6,)) # The 6-feature vector

x = tf.keras.layers.Concatenate()([lat_input, cond_input])
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(2706 * 13)(x) # Output logits for every node/district
x = tf.keras.layers.Reshape((2706, 13))(x)
# Softmax over the 13 districts for each node
dec_output = tf.keras.layers.Softmax(axis=-1)(x)

decoder = tf.keras.Model([lat_input, cond_input], dec_output, name="decoder")

# --- FULL CVAE ---
cvae_input_map = tf.keras.Input(shape=(2706, 13))
cvae_input_metrics = tf.keras.Input(shape=(6,))

z_m, z_lv, z_sample, m_pred = encoder(cvae_input_map)
reconstruction = decoder([z_sample, cvae_input_metrics])

cvae = tf.keras.Model([cvae_input_map, cvae_input_metrics], 
                      [reconstruction, z_m, z_lv, m_pred], 
                      name="cvae")

# 4. CUSTOM LOSS FUNCTION
# ---------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def compute_loss(map_true, metric_true, map_pred, z_mean, z_log_var, metric_pred):
    # 1. Reconstruction Loss (Categorical Crossentropy)
    # How well does the map match the input?
    recon_loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(map_true, map_pred)
    ) * 2706 # Scale by number of nodes
    
    # 2. KL Divergence (Regularization)
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    
    # 3. Metric Prediction Loss (Auxiliary)
    # Does the latent space actually capture the metrics?
    met_loss = tf.reduce_mean(tf.square(metric_true - metric_pred))
    
    # 4. CONTIGUITY LOSS (The "Smoothness" Term)
    # We want neighbors to have the same district assignment.
    # Dot product of a node's distribution with its neighbors' distributions.
    # Maximize (pred * neighbors) => Minimize -(pred * neighbors)
    
    # neighbor_sums = A * pred (Shape: Batch, Nodes, 13)
    neighbor_sums = tf.einsum('nm,bmu->bnu', ADJ_RAW, map_pred)
    
    # Dot product: sum(pred_i * pred_neighbors_i)
    # This value is high if a node and its neighbors share the same district
    smoothness = tf.reduce_sum(map_pred * neighbor_sums, axis=[1, 2])
    contiguity_loss = -tf.reduce_mean(smoothness)
    
    # Weights for the terms (You can tune these!)
    total_loss = recon_loss + (1.5 * kl_loss) + (10.0 * met_loss) + (0.01 * contiguity_loss)
    
    return total_loss, recon_loss, contiguity_loss

# 5. TRAINING LOOP
# ---------------------------------------------------------
print("Starting Training...")
for epoch in range(EPOCHS):
    total_loss_tracker = 0
    recon_tracker = 0
    contig_tracker = 0
    steps = 0
    
    for batch_maps, batch_metrics in train_dataset:
        with tf.GradientTape() as tape:
            # Forward pass
            map_pred, z_m, z_lv, m_pred = cvae([batch_maps, batch_metrics])
            # Calculate loss
            loss, rc, ct = compute_loss(batch_maps, batch_metrics, map_pred, z_m, z_lv, m_pred)
            
        # Backward pass
        grads = tape.gradient(loss, cvae.trainable_weights)
        optimizer.apply_gradients(zip(grads, cvae.trainable_weights))
        
        total_loss_tracker += loss
        recon_tracker += rc
        contig_tracker += ct
        steps += 1
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss_tracker/steps:.2f} | Recon: {recon_tracker/steps:.2f} | Contig: {contig_tracker/steps:.2f}")

# 6. SAVE
print("Saving Model...")
cvae.save("model_v2_full.h5")
# Save encoder/decoder separately for easier inference later
encoder.save("model_v2_encoder.h5")
decoder.save("model_v2_decoder.h5")
print("Done!")