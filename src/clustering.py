from sklearn.cluster import KMeans

def cluster_regions(df, features, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = model.fit_predict(df[features])
    return df
