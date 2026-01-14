import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

# -------------------------------
# Load cleaned data
# -------------------------------
df = pd.read_csv("data/cleaned_data.csv")
print("Cleaned data shape:", df.shape)

# -------------------------------
# Feature selection
# -------------------------------
features = df[['city', 'cuisine', 'rating', 'rating_count', 'cost']]

# -------------------------------
# One Hot Encoding
# -------------------------------
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

encoded_cat = encoder.fit_transform(features[['city', 'cuisine']])

encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(['city', 'cuisine']),
    index=features.index
)

# -------------------------------
# Combine numerical + encoded
# -------------------------------
numerical_df = features[['rating', 'rating_count', 'cost']]
encoded_df = pd.concat([numerical_df, encoded_cat_df], axis=1)

print("Encoded data shape:", encoded_df.shape)

# -------------------------------
# Save encoded data (FAST)
# -------------------------------
encoded_df.to_pickle("data/encoded_data.pkl")

# Save encoder
with open("models/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# -------------------------------
# Alignment check
# -------------------------------
assert len(df) == len(encoded_df)

print("✅ Encoding completed successfully")
