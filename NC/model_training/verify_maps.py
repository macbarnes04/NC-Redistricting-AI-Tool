import tensorflow as tf
import numpy as np
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = "model_v2_decoder.h5"
DATA_PATH = "NC/data/processed_v2_data_1k.npz"
SHAPEFILE_PATH = "NC/NC_precs/NC_precs_all_data.shp" # Must match your training data source
OUTPUT_FOLDER = "generated_maps"
LATENT_DIM = 64
NUM_DISTRICTS = 13

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. LOAD DATA
# ---------------------------------------------------------
print("Loading Adjacency Matrix...")
data = np.load(DATA_PATH)
adj_matrix = data['adjacency']

# Build NetworkX Graph
G = nx.from_numpy_array(adj_matrix)

print("Loading Shapefile...")
gdf = gpd.read_file(SHAPEFILE_PATH)

print("Loading Decoder Model...")
decoder = tf.keras.models.load_model(MODEL_PATH)

# Setup consistent colormap (Tab20 has 20 distinct colors, good for 13 districts)
base_cmap = plt.get_cmap('tab20')
# Create a fixed dictionary: District ID -> RGBA Color
DISTRICT_COLORS = {i: base_cmap(i % 20) for i in range(NUM_DISTRICTS)}

# 3. GENERATE MAPS
# ---------------------------------------------------------
print("\n--- GENERATING 5 RANDOM MAPS ---")
random_z = tf.random.normal(shape=(5, LATENT_DIM))
dummy_metrics = tf.zeros(shape=(5, 6))

generated_probs = decoder.predict([random_z, dummy_metrics])
generated_maps = np.argmax(generated_probs, axis=-1)

# 4. ANALYZE & VISUALIZE
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

for i in range(5):
    print(f"\nProcessing Map {i+1}...")
    current_assignment = generated_maps[i]
    
    # Store analysis results for the table
    # Format: (District ID, Piece Count, Status String)
    analysis_results = []
    total_broken = 0
    
    # --- A. CHECK CONTIGUITY ---
    for d_id in range(NUM_DISTRICTS):
        nodes_in_dist = [n for n, dist in enumerate(current_assignment) if dist == d_id]
        
        if len(nodes_in_dist) == 0:
            analysis_results.append((d_id, 0, "EMPTY"))
            continue

        subgraph = G.subgraph(nodes_in_dist)
        
        if nx.is_connected(subgraph):
            analysis_results.append((d_id, 1, "Valid"))
        else:
            pieces = nx.number_connected_components(subgraph)
            analysis_results.append((d_id, pieces, f"BROKEN ({pieces})"))
            total_broken += 1

    overall_status = "VALID MAP" if total_broken == 0 else f"FAILED ({total_broken} dists broken)"
    print(f"  RESULT: {overall_status}")

    # --- B. VISUALIZE WITH TABLE ---
    print(f"  Generating layout...")
    
    # 1. Assign colors to the GeoDataFrame based on our fixed dictionary
    # We map the predicted ID to the specific RGBA color
    gdf['district_id'] = current_assignment
    gdf['color'] = gdf['district_id'].map(DISTRICT_COLORS)
    
    # 2. Setup Figure with GridSpec (Map on left, Table on right)
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    
    ax_map = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    
    # 3. Plot Map
    gdf.plot(color=gdf['color'], 
             ax=ax_map, 
             edgecolor='black', 
             linewidth=0.1)
    ax_map.set_axis_off()
    ax_map.set_title(f"AI Generated Map #{i+1}\n{overall_status} | {timestamp}", fontsize=14)
    
    # 4. Create Table Data
    table_cell_text = []
    table_cell_colors = []
    
    headers = ["Dist", "Pieces / Status"]
    
    for d_id, pieces, status_str in analysis_results:
        # Text for the row
        row_text = [f"#{d_id+1}", status_str] # Display as 1-13 instead of 0-12
        table_cell_text.append(row_text)
        
        # Colors: Left cell is District Color, Right cell is status color
        # If broken, make the status cell light red
        status_bg = "#ffcccc" if pieces > 1 else "#ccffcc" 
        if pieces == 0: status_bg = "#eeeeee" # Grey for empty
        
        row_colors = [DISTRICT_COLORS[d_id], status_bg]
        table_cell_colors.append(row_colors)

    # 5. Draw Table
    ax_table.axis('off')
    the_table = ax_table.table(
        cellText=table_cell_text,
        cellColours=table_cell_colors,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    the_table.scale(1, 2) # Make rows taller
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # Save
    filename = f"map_{i+1}_analyzed_{datetime.now().strftime('%H%M%S')}.png"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

print(f"\nDone! Images saved to '{OUTPUT_FOLDER}'")