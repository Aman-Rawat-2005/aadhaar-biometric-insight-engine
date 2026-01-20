from sklearn.ensemble import IsolationForest

def detect_anomalies(df, feature):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[[feature]])
    return df
