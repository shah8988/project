from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X, labels)
    }