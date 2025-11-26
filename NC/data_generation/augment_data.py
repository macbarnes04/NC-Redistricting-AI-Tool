import json
import numpy as np
import os
from gerrychain import Graph, Partition, Election, updaters
from gerrychain.metrics import efficiency_gap


# Make sure the path matches where your shapefile is
g = Graph.from_file("NC/NC_precs/NC_precs_all_data.shp")
print("COLUMNS FOUND:")
print(sorted(list(g.nodes[0].keys()))) # print out data column heads 

# 1. SETUP: Define paths
# ---------------------------------------------------------
SHAPEFILE_PATH = "NC/NC_precs/NC_precs_all_data.shp"
OUTPUT_PATH = "processed_v2_data.npz"

# Logic to switch between full 10k dataset and sample dataset
if os.path.exists("NC/maps/10KPlansFinal.json"):
    EXISTING_PLANS_PATH = "NC/maps/10K_Plans.json"
    print("Using FULL Dataset (10k maps)")
else:
    # Make sure you renamed your 1000-map file to this!
    EXISTING_PLANS_PATH = "NC/data/sample_maps.json"
    print("WARNING: Full dataset not found. Using SAMPLE Dataset")
# ---------------------------------------------------------

print("Loading Graph from shapefile...")
# Load the graph with the underlying data
graph = Graph.from_file(SHAPEFILE_PATH)

# DEBUG: Print columns to help you verify names if needed
# print("Available Data Columns:", list(graph.nodes[0].keys()))

# --- USER CONFIGURATION: SET CORRECT COLUMN NAMES HERE ---
POP_COL = "tot"        # Total Population
# Based on your previous files, this is likely 'BPOP' or 'BVAP'. 
# Check the print output if it crashes!
BLACK_POP_COL = "bla_alo" 
# -------------------------------------------------------

# 2. SETUP: Define Updaters
my_updaters = {
    "population": updaters.Tally(POP_COL, alias="population"),
    "black_pop": updaters.Tally(BLACK_POP_COL, alias="black_pop"),
}

# Add Election updaters (using 2016 Senate race as proxy)
# Columns from your original script: Dem="G16USSDROS", Rep="G16USSRBUR"
election = Election("Senate16", {"Dem": "G16USSDROS", "Rep": "G16USSRBUR"})
my_updaters["Senate16"] = election

print(f"Loading plans from {EXISTING_PLANS_PATH}...")
with open(EXISTING_PLANS_PATH, "r") as f:
    raw_data = json.load(f)

# The assignments are stored as a list of lists
assignments_list = raw_data["EquivalencyFiles"] 

# IMPORTANT: We also grab the County Splits directly from the JSON
# (No need to recalculate this expensive metric!)
county_splits_list = raw_data["CountySplitsScores"]

new_features_list = []
valid_maps_list = []

print(f"Processing {len(assignments_list)} maps to calculate 6-feature vector...")

for i, assignment in enumerate(assignments_list):
    # Convert list to dict mapping {node_id: district_id}
    # Assuming list order matches graph.nodes order
    assignment_dict = {node: district for node, district in enumerate(assignment)}
    
    # Create a Partition
    part = Partition(graph, assignment_dict, my_updaters)
    
    # --- METRIC 1: COMPACTNESS (Cut Edges) ---
    cut_edges_val = len(part["cut_edges"])
    
    # --- METRIC 2: GEOGRAPHY (County Splits) ---
    # We pull this directly from your pre-calculated list
    splits_val = county_splits_list[i]
    
    # --- METRIC 3: FAIRNESS (Mean-Median) ---
    mean_median_val = part["Senate16"].mean_median()
    
    # --- METRIC 4: FAIRNESS (Efficiency Gap) ---
    eff_gap_val = part["Senate16"].efficiency_gap()
    
    # --- METRIC 5: PARTISAN OUTCOME (Democratic Seats) ---
    # .wins("Dem") returns the integer number of districts won by Democrats
    dem_seats_val = part["Senate16"].wins("Dem")
    
    # --- METRIC 6: RACIAL EQUITY (Black Pop Variance) ---
    # Get % Black Population in every district
    dist_pops = part["population"]
    dist_black_pops = part["black_pop"]
    
    black_percents = []
    for dist_id in dist_pops.keys():
        total = dist_pops[dist_id]
        if total > 0:
            black_percents.append(dist_black_pops[dist_id] / total)
        else:
            black_percents.append(0)
            
    # Calculate Variance (Standard Deviation) of these percentages
    black_pop_std = np.std(black_percents)
    
    # FINAL VECTOR (6 Dimensions)
    # [CutEdges, CountySplits, MeanMedian, EffGap, DemSeats, BlackPopStd]
    new_features_list.append([
        cut_edges_val, 
        splits_val, 
        mean_median_val, 
        eff_gap_val, 
        dem_seats_val, 
        black_pop_std
    ])
    
    # Keep the map assignment for training
    valid_maps_list.append(assignment)

    if i % 100 == 0:
        print(f"Processed {i} maps...")

# 3. SAVE EVERYTHING
# We save the Adjacency Matrix too, because the GNN needs it!
print("Building Adjacency Matrix...")
adj_matrix = np.zeros((len(graph), len(graph)), dtype=np.float32)
for node in graph.nodes:
    for neighbor in graph.neighbors(node):
        adj_matrix[node][neighbor] = 1.0

print(f"Saving to {OUTPUT_PATH}...")
np.savez_compressed(
    OUTPUT_PATH,
    maps=np.array(valid_maps_list, dtype=np.int32),
    features=np.array(new_features_list, dtype=np.float32),
    adjacency=adj_matrix
)
print("Done! You are ready for V2.0 training.")