import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# --------------------------------
# App Config
# --------------------------------
st.set_page_config(
    page_title="Swiggy Restaurant Recommendation",
    layout="wide"
)

st.title("🍽️ Swiggy Restaurant Recommendation System")
st.write("Select a restaurant and get similar recommendations")

# --------------------------------
# Load Data
# --------------------------------
@st.cache_data
def load_data():
    cleaned_df = pd.read_csv("data/cleaned_data.csv")
    encoded_df = pd.read_pickle("data/encoded_data.pkl")
    encoded_df = encoded_df.fillna(0)
    return cleaned_df, encoded_df

cleaned_df, encoded_df = load_data()

# --------------------------------
# Train Nearest Neighbors
# --------------------------------
@st.cache_resource
def train_model(encoded_data):
    model = NearestNeighbors(
        n_neighbors=6,
        metric="cosine",
        algorithm="brute"
    )
    model.fit(encoded_data)
    return model

nn_model = train_model(encoded_df.values)

# --------------------------------
# Sidebar Filters
# --------------------------------
st.sidebar.header("🔍 Filters")

city = st.sidebar.selectbox(
    "Select City",
    sorted(cleaned_df["city"].unique())
)

filtered_df = cleaned_df[cleaned_df["city"] == city]

restaurant_name = st.sidebar.selectbox(
    "Select Restaurant",
    sorted(filtered_df["name"].unique())
)

# --------------------------------
# Recommendation Logic
# --------------------------------
def recommend_restaurants(restaurant_name, city, top_n=5):
    idx = cleaned_df[
        (cleaned_df["name"] == restaurant_name) &
        (cleaned_df["city"] == city)
    ].index[0]

    distances, indices = nn_model.kneighbors(
        encoded_df.iloc[idx].values.reshape(1, -1)
    )

    rec_indices = indices[0][1:top_n + 1]

    return cleaned_df.iloc[rec_indices]

# --------------------------------
# Display Recommendations
# --------------------------------
if st.sidebar.button("🍴 Recommend"):
    st.subheader("✨ Recommended Restaurants")

    recommendations = recommend_restaurants(
        restaurant_name,
        city
    )

    for _, row in recommendations.iterrows():
        st.markdown(f"""
        ### 🍽️ {row['name']}
        - **Cuisine:** {row['cuisine']}
        - **Rating:** ⭐ {row['rating']}
        - **Cost for Two:** ₹{row['cost']}
        - **Address:** {row['address']}
        - [View on Swiggy]({row['link']})
        ---
        """)
