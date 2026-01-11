from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_vae(latent, k=5):
    return KMeans(n_clusters=k, random_state=42).fit_predict(latent)

def cluster_pca(features, k=5, dim=16):
    X_pca = PCA(n_components=dim).fit_transform(features)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_pca)
    return X_pca, labels