import numpy as np
import tensorflow as tf
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# CONFIGURATION
MODEL_PATH = "model_v2_decoder.h5" # Or your checkpoint path
DATA_PATH = "NC/data/processed_v2_data_10k.npz"
SHAPEFILE_PATH = "NC/NC_precs/NC_precs_all_data.shp"
OUTPUT_FOLDER = "repaired_maps"
LATENT_DIM = 64
NUM_DISTRICTS = 13

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. LOAD DATA & MODEL
print("Loading Data...")
data = np.load(DATA_PATH)
adj_matrix = data['adjacency']
G = nx.from_numpy_array(adj_matrix)
gdf = gpd.read_file(SHAPEFILE_PATH)
decoder = tf.keras.models.load_model(MODEL_PATH)

# Colors
base_cmap = plt.get_cmap('tab20')
DISTRICT_COLORS = {i: base_cmap(i % 20) for i in range(NUM_DISTRICTS)}

# 2. GENERATE RAW MAP
print("Generating a map to repair...")
# We use a fixed seed to reproduce your error or a random one
random_z = tf.random.normal(shape=(1, LATENT_DIM))
dummy_metrics = tf.zeros(shape=(1, 6))

# Get PROBABILITIES (Softmax), not just the hard choice
probs = decoder.predict([random_z, dummy_metrics])[0] # Shape: (2706, 13)
initial_assignment = np.argmax(probs, axis=1)

# 3. REPAIR FUNCTION
def repair_map(assignment, probabilities, graph):
    """
    1. Fix Empty Districts (Resurrection)
    2. Fix Disconnected Components (Island Merging)
    """
    fixed_assignment = assignment.copy()
    
    # --- STEP 1: RESURRECTION (Fix Empty Districts) ---
    # Check which districts are missing
    present_districts = set(fixed_assignment)
    missing_districts = [d for d in range(NUM_DISTRICTS) if d not in present_districts]
    
    for missing_d in missing_districts:
        print(f"  ⚠ Resurrection: Reviving District {missing_d}...")
        # Find the node that had the HIGHEST probability of being this missing district
        # (even if it lost the vote to someone else)
        # We look at the probability column for the missing district
        best_candidate_node = np.argmax(probabilities[:, missing_d])
        
        # Force this node to be the missing district
        fixed_assignment[best_candidate_node] = missing_d
        print(f"    - Seeded District {missing_d} at Node {best_candidate_node}")

    # --- STEP 2: ISLAND MERGING ---
    # We loop until everything is valid
    max_cycles = 10
    for cycle in range(max_cycles):
        print(f"  Cycle {cycle+1}: Merging islands...")
        changes_made = 0
        
        for d_id in range(NUM_DISTRICTS):
            nodes_in_dist = [n for n, d in enumerate(fixed_assignment) if d == d_id]
            if not nodes_in_dist: continue

            subgraph = graph.subgraph(nodes_in_dist)
            if nx.is_connected(subgraph):
                continue
                
            # It's broken. Find the largest component (The "Mainland")
            components = list(nx.connected_components(subgraph))
            components.sort(key=len, reverse=True)
            mainland = components[0]
            islands = components[1:]
            
            # For every island, flip it to the most common neighbor
            for island in islands:
                for node in island:
                    # Find neighbors of this node
                    neighbors = list(graph.neighbors(node))
                    neighbor_districts = [fixed_assignment[n] for n in neighbors]
                    
                    # Find most common neighbor district that ISN'T the broken one
                    # (We want to merge into a DIFFERENT district)
                    valid_neighbors = [d for d in neighbor_districts if d != d_id]
                    
                    if valid_neighbors:
                        # Pick the most common valid neighbor
                        new_dist = max(set(valid_neighbors), key=valid_neighbors.count)
                        fixed_assignment[node] = new_dist
                        changes_made += 1
        
        if changes_made == 0:
            print("  No more islands found!")
            break
            
    return fixed_assignment

# 4. RUN REPAIR
print("\n--- REPAIRING MAP ---")
final_map = repair_map(initial_assignment, probs, G)

# 5. VERIFY FINAL STATUS
broken_count = 0
for d_id in range(NUM_DISTRICTS):
    nodes = [n for n, d in enumerate(final_map) if d == d_id]
    if len(nodes) == 0:
        print(f"District {d_id}: STILL EMPTY")
        broken_count += 1
        continue
    sub = G.subgraph(nodes)
    if not nx.is_connected(sub):
        print(f"District {d_id}: STILL BROKEN")
        broken_count += 1

status = "SUCCESS" if broken_count == 0 else "PARTIAL FAIL"
print(f"Final Status: {status}")

# 6. VISUALIZE
print("Saving image...")
gdf['district'] = final_map
gdf['color'] = gdf['district'].map(DISTRICT_COLORS)

fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(color=gdf['color'], ax=ax, edgecolor='black', linewidth=0.1)
plt.title(f"Repaired AI Map\nStatus: {status}")
plt.axis('off')
plt.savefig(f"{OUTPUT_FOLDER}/repaired_map.png", dpi=150)
print("Done!")