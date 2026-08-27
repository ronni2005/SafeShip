"""
Train/Test Split

80/20 split, stratified on 'rejected' so both sets have a similar
reject-rate mix (avoids a test set that's accidentally all-safe or
all-risky by chance).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/orders.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["rejected"]  # keeps reject % consistent across both sets
)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"Train: {len(train_df)} rows | reject rate: {train_df['rejected'].mean():.1%}")
print(f"Test:  {len(test_df)} rows | reject rate: {test_df['rejected'].mean():.1%}")
