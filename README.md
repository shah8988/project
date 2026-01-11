
Unsupervised VAE-based Clustering of Hybrid Music Data

Dataset: FUTGA MusicCaps
Method: Variational Autoencoder + KMeans
Baseline: PCA + KMeans

Features:
- Audio MFCC
- Text caption embeddings

Metrics:
- Silhouette Score
- Calinski-Harabasz Index

This project implements a Variational Autoencoder (VAE) using the MusicCaps dataset captions.

- src/: source code
- notebooks/: main and exploratory notebooks
- data/processed/: extracted features
- results/plots/: t-SNE visualizations
