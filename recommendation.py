import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --------------------------------
# Load data
# --------------------------------
encoded_df = pd.read_pickle("data/encoded_data.pkl")
cleaned_df = pd.read_csv("data/cleaned_data.csv")

print("Encoded data shape:", encoded_df.shape)
print("Cleaned data shape:", cleaned_df.shape)

# --------------------------------
# Handle NaN values
# --------------------------------
encoded_df = encoded_df.fillna(0)

# --------------------------------
# Fit Nearest Neighbors model
# --------------------------------
print("Training Nearest Neighbors model...")

nn_model = NearestNeighbors(
    n_neighbors=6,   # 1 itself + 5 recommendations
    metric="cosine",
    algorithm="brute"
)

nn_model.fit(encoded_df.values)

print("✅ Model trained")

# --------------------------------
# Recommendation function
# --------------------------------
def recommend_restaurants(index, top_n=5):
    query_vector = encoded_df.iloc[index].values.reshape(1, -1)

    distances, indices = nn_model.kneighbors(query_vector)

    recommended_indices = indices[0][1:top_n + 1]

    return cleaned_df.iloc[recommended_indices][
        ['name', 'city', 'cuisine', 'rating', 'cost', 'address', 'link']
    ]

# --------------------------------
# Test
# --------------------------------
if __name__ == "__main__":
    sample_index = 10
    print("\nRecommended Restaurants:")
    print(recommend_restaurants(sample_index))
