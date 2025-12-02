import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import os
import gdown
import zipfile

# ---------------------------------------------------------
# 1. CONFIGURATION & CLOUD SETUP
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="NC Redistricting AI")

# Google Drive File IDs (From your Colab Notebook)
FILE_IDS = {
    "model": "1sJeVHP6_OcGdO2sTLLmIA5MGHKzt-qRl",
    "data": "1TU405s2QIQNxOuxksPMRoQpK4WTRe4yh",
    "shapefile": "1vD3oKO1WaQ1I1hrpcXgUjvdrUrq_VHgi"
}

# Local Paths
BASE_DIR = "."
MODEL_PATH = os.path.join(BASE_DIR, "model_v2_decoder.h5")
DATA_PATH = os.path.join(BASE_DIR, "processed_v2_data_10k.npz")
SHAPEFILE_ZIP = os.path.join(BASE_DIR, "NC_precs.zip")
SHAPEFILE_DIR = os.path.join(BASE_DIR, "NC_precs")
SHAPEFILE_PATH = os.path.join(SHAPEFILE_DIR, "NC_precs_all_data.shp")

# --- DOWNLOADER FUNCTION ---
@st.cache_resource
def setup_files():
    """Downloads files from Google Drive if they don't exist."""
    
    # 1. Download Model
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_IDS["model"]}'
        gdown.download(url, MODEL_PATH, quiet=False)
        
    # 2. Download Data
    if not os.path.exists(DATA_PATH):
        url = f'https://drive.google.com/uc?id={FILE_IDS["data"]}'
        gdown.download(url, DATA_PATH, quiet=False)

    # 3. Download & Unzip Shapefile
    if not os.path.exists(SHAPEFILE_PATH):
        if not os.path.exists(SHAPEFILE_ZIP):
            url = f'https://drive.google.com/uc?id={FILE_IDS["shapefile"]}'
            gdown.download(url, SHAPEFILE_ZIP, quiet=False)
        
        # Unzip
        if os.path.exists(SHAPEFILE_ZIP):
            with zipfile.ZipFile(SHAPEFILE_ZIP, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)

# Run Setup immediately
with st.spinner("Downloading model assets from cloud storage... (This happens once)"):
    setup_files()

# ---------------------------------------------------------
# 2. APP CONSTANTS
# ---------------------------------------------------------
NUM_DISTRICTS = 13
LATENT_DIM = 64
FEATURE_NAMES = [
    "Compactness (Cut Edges)",
    "County Splits",
    "Mean-Median Score",
    "Efficiency Gap",
    "Democratic Seats",
    "Black Pop Variance"
]

# ---------------------------------------------------------
# 3. CACHED LOADING FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the Keras Decoder Model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_training_stats():
    """Loads training data to get means and std devs for sliders."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data not found at {DATA_PATH}")
        return None, None, None, None
        
    data = np.load(DATA_PATH)
    features = data['features']
    adj = data['adjacency']
    
    feat_min = np.min(features, axis=0)
    feat_max = np.max(features, axis=0)
    feat_mean = np.mean(features, axis=0)
    
    return feat_min, feat_max, feat_mean, adj

@st.cache_data
def load_shapefile():
    """Loads the precinct shapefile."""
    if not os.path.exists(SHAPEFILE_PATH):
        st.error(f"Shapefile not found at {SHAPEFILE_PATH}")
        return None
    return gpd.read_file(SHAPEFILE_PATH)

# ---------------------------------------------------------
# 4. REPAIR LOGIC
# ---------------------------------------------------------
def repair_map_advanced(assignment, graph, probabilities):
    """Fixes islands and missing districts."""
    fixed_assignment = assignment.copy()
    
    # Phase 1: Resurrection
    present_districts = set(fixed_assignment)
    missing_districts = [d for d in range(NUM_DISTRICTS) if d not in present_districts]
    
    for missing_d in missing_districts:
        best_candidate_node = np.argmax(probabilities[:, missing_d])
        fixed_assignment[best_candidate_node] = missing_d
    
    # Phase 2: Island Merging
    max_cycles = 5
    for cycle in range(max_cycles):
        changes_made = 0
        for d_id in range(NUM_DISTRICTS):
            nodes_in_dist = [n for n, d in enumerate(fixed_assignment) if d == d_id]
            if not nodes_in_dist: continue

            subgraph = graph.subgraph(nodes_in_dist)
            if nx.is_connected(subgraph):
                continue
            
            components = list(nx.connected_components(subgraph))
            components.sort(key=len, reverse=True)
            islands = components[1:]
            
            for island in islands:
                for node in island:
                    neighbors = list(graph.neighbors(node))
                    neighbor_dists = [fixed_assignment[n] for n in neighbors]
                    valid_neighbors = [d for d in neighbor_dists if d != d_id]
                    
                    if valid_neighbors:
                        new_dist = max(set(valid_neighbors), key=valid_neighbors.count)
                        fixed_assignment[node] = new_dist
                        changes_made += 1
        if changes_made == 0:
            break
            
    return fixed_assignment

# ---------------------------------------------------------
# 5. METRIC CALCULATION
# ---------------------------------------------------------
def calculate_actual_metrics(gdf, assignment_col):
    """Calculates actual partisan/demographic stats."""
    # Group by district
    districts = gdf.groupby(assignment_col).sum(numeric_only=True)
    
    # Calculate Seats
    districts['Dem_Votes'] = districts['G16USSDROS']
    districts['Rep_Votes'] = districts['G16USSRBUR']
    districts['Winner'] = np.where(districts['Dem_Votes'] > districts['Rep_Votes'], 'D', 'R')
    dem_seats = districts[districts['Winner'] == 'D'].shape[0]
    
    # Efficiency Gap
    districts['D_Wasted'] = np.where(
        districts['Winner'] == 'D', 
        districts['Dem_Votes'] - (districts['Dem_Votes'] + districts['Rep_Votes'])/2, 
        districts['Dem_Votes']
    )
    districts['R_Wasted'] = np.where(
        districts['Winner'] == 'R', 
        districts['Rep_Votes'] - (districts['Dem_Votes'] + districts['Rep_Votes'])/2, 
        districts['Rep_Votes']
    )
    total_votes = districts['Dem_Votes'].sum() + districts['Rep_Votes'].sum()
    eff_gap = (districts['D_Wasted'].sum() - districts['R_Wasted'].sum()) / total_votes
    
    return dem_seats, eff_gap, districts

# ---------------------------------------------------------
# 6. MAIN APP UI
# ---------------------------------------------------------
def main():
    st.title("NC AI Redistricting Lab 🤖🗺️")
    st.markdown("Explore how tweaking fairness metrics impacts the political landscape of North Carolina.")

    # Load Resources
    model = load_model()
    feat_min, feat_max, feat_mean, adj_matrix = load_training_stats()
    gdf = load_shapefile()

    if model is None or feat_mean is None or gdf is None:
        st.stop() # Stop execution if files failed to load

    # Build Graph for Repair
    G = nx.from_numpy_array(adj_matrix)

    # --- SIDEBAR: INPUTS ---
    st.sidebar.header("🎯 Set Target Metrics")
    
    user_conditions = []
    
    for i, name in enumerate(FEATURE_NAMES):
        min_val = float(feat_min[i])
        max_val = float(feat_max[i])
        default_val = float(feat_mean[i])
        range_pad = (max_val - min_val) * 0.1
        
        val = st.sidebar.slider(
            label=name,
            min_value=min_val - range_pad,
            max_value=max_val + range_pad,
            value=default_val
        )
        user_conditions.append(val)

    user_cond_vector = np.array([user_conditions])

    # --- MAIN AREA ---
    if st.sidebar.button("Generate Map", type="primary"):
        with st.spinner("Dreaming up a new map... (AI Inference)"):
            noise = tf.random.normal(shape=(1, LATENT_DIM))
            probs = model.predict([noise, user_cond_vector], verbose=0)[0]
            raw_assignment = np.argmax(probs, axis=1)
            
        with st.spinner("Fixing border irregularities... (Graph Repair)"):
            final_assignment = repair_map_advanced(raw_assignment, G, probs)
            
        # Visualize
        gdf['District'] = final_assignment
        dem_seats, eff_gap, dist_stats = calculate_actual_metrics(gdf, 'District')
        
        # Partisan Coloring
        dist_stats['Margin'] = (dist_stats['Dem_Votes'] - dist_stats['Rep_Votes']) / (dist_stats['Dem_Votes'] + dist_stats['Rep_Votes'])
        gdf['Vote_Margin'] = gdf['District'].map(dist_stats['Margin'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        gdf.plot(column='Vote_Margin', 
                 cmap='coolwarm_r', 
                 ax=ax, 
                 edgecolor='white', 
                 linewidth=0.2,
                 vmin=-0.3, vmax=0.3)
        ax.axis('off')
        st.pyplot(fig)
        
        # Metrics Table
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Results Analysis")
            st.metric("Actual Democratic Seats", f"{dem_seats} / {NUM_DISTRICTS}", 
                      delta=f"{dem_seats - user_conditions[4]:.1f} from target")
            st.metric("Actual Efficiency Gap", f"{eff_gap:.3f}", 
                      delta=f"{eff_gap - user_conditions[3]:.3f} from target")
            
        with col2:
            st.subheader("District Breakdown")
            display_df = dist_stats[['Dem_Votes', 'Rep_Votes', 'Winner']].copy()
            display_df['Margin'] = dist_stats['Margin'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_df)

        # Comparison
        st.subheader("Input vs. Output Comparison")
        comparison_data = {
            "Metric": FEATURE_NAMES,
            "User Target": user_conditions,
            "Actual Map": ["-", "-", "-", f"{eff_gap:.3f}", dem_seats, "-"] 
        }
        st.table(pd.DataFrame(comparison_data))

if __name__ == "__main__":
    main()