import pandas as pd

df = pd.read_csv("mudra_landmarks.csv")

print("Total samples:", len(df))
print("\nClass count:")
print(df["label"].value_counts())
