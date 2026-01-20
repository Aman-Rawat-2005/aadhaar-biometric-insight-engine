def generate_rules(df):
    rules = []

    for _, row in df.iterrows():
        if row["growth_rate"] > 0.5:
            rules.append("High biometric transition pressure detected.")
        elif row["BRS"] < 0.4:
            rules.append("Low biometric resilience – intervention needed.")
        else:
            rules.append("Biometric system stable.")

    df["decision_note"] = rules
    return df
