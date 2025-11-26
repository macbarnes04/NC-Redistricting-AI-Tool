# 🗺️ AI-Driven Redistricting for North Carolina (V2.0) 🗺️
### Author: Mac Barnes
---

### 📌 Project Overview

This project audits and improves an Automated Decision System (ADS) designed to generate congressional district maps for North Carolina. Specifically, it investigates whether Generative AI (using Graph Neural Networks and Variational Autoencoders) can be used as a "pro-democracy" tool to counter authoritarian gerrymandering.

I built a V2.0 Model that improves upon my [original 2021 paper](https://drive.google.com/file/d/10fSQ548zBNBA5uEeO5kb44-7K02Ld-we/view) by:

- Enforcing Contiguity: Using Graph Convolutional Networks (GCNs) to understand spatial relationships.
- Increasing Control: Conditioning generation on a 6-dimensional feature vector (Compactness, Efficiency Gap, Seat Count, Racial Variance, etc.).
- Visual Auditing: Testing the model for intrinsic bias, partisan fidelity, and robustness.

---

### 📂 Repository Structure
```
NC-Redistricting-AI-Tool/
├── NC/
│   ├── data_generation/    # Scripts to create training data
│   │   ├── augment_data.py # Adds new metrics (Eff Gap, Race) to existing maps
│   │   └── GerryChain...   # Original MCMC generation script
│   ├── model_training/     # Model architecture and training loops
│   │   ├── model_v2.py     # The NEW GNN-CVAE model (V2.0)
│   │   └── model_v1.py # The OLD model (V1.0 reference)
│   ├── NC_precs/           # Shapefiles for North Carolina precincts
│   └── maps/               # Storage for generated map data
├── utils/                  # Helper functions for plotting and metrics
└── README.md               # This file
```

---

### 📊 Data Access

Due to GitHub file size limits (>100MB), the full training dataset is not hosted in this repository. But you have the code to generate it locally.

Sample Data: A smaller sample dataset (sample_maps.json) is included in NC/maps/ for testing the code.
Full Dataset: The full 10,000-map ensemble and trained model weights (model_v2_decoder.h5) can be downloaded from this [Google Drive Folder](https://drive.google.com/drive/folders/1srJTpnubUYqd9Jh9XkE2tM9gzqLpkcIz?usp=drive_link).

---

### 🛠️ Methodology

**1. Data Generation (MCMC)**

GerryChain script generates an ensemble of 10,000 random-walk maps. This serves as our "neutral baseline" of the state's political geography.

**2. Model Architecture (GNN-CVAE)**

Implemented a Conditional Variational Autoencoder with a GNN Encoder.

Input: Map Assignment (One-Hot) + Adjacency Matrix (Graph).

Condition: 6-Feature Vector (Cut Edges, County Splits, Mean-Median, Efficiency Gap, Dem Seats, Black Pop Variance).

Loss Function: Reconstruction Loss + KL Divergence + Contiguity Loss (penalizing neighbor disagreements).

**3. Auditing**

The model was audited for:
- Technical Validity: Can it draw contiguous maps? (Yes, with minor post-processing).
- Fidelity: Does it obey user commands? (High fidelity for geometry/race, low fidelity for specific seat counts).
- Fairness: Does it introduce bias? (No, it mirrors the training distribution perfectly).

--- 

### 🚀 Additional testing via Audit (Google Colab)

You can replicate the full audit, including map generation and fairness analysis, using the [Google Colab notebook](https://colab.research.google.com/drive/1-_slfTugyfk8JbdSSYK2OETQwwF4oTPO?usp=sharing).

Note: To run the notebook successfully, you may need to download the data files (see below) and upload them to your Google Drive or use the public download links provided in the notebook (suggested).

---

### Acknowledgements
- Original Paper sponsored via the [NCSSM Research in Computational Sciences Program](https://research-innovation.ncssm.edu/departmental-programs/durham-campus/research-in-computational-science) under [Bob Gotwals](https://www.ncssm.edu/directory/bob-gotwals)
- Original Paper and methodology written in coordination with the [Princeton Gerrymandering Project](https://gerrymander.princeton.edu/)
- Audit Contributors: Mike Tan
- Supervised By: [Athena Tabhaki](https://engineering.washu.edu/faculty/Athena-Tabakhi.html), WashU CSE Department via CSE 3050 - Responsible Data Science
