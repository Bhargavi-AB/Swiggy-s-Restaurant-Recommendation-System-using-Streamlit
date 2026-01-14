import pandas as pd

# -------------------------------
# STEP 1: Load the raw dataset
# -------------------------------
df = pd.read_csv("C:/Users/Bhargavi/OneDrive/Desktop/swiggy_recommendation/data/swiggy.csv")

print("Initial dataset shape:", df.shape)

# -------------------------------
# STEP 2: Remove duplicate rows
# -------------------------------
df.drop_duplicates(inplace=True)

print("After removing duplicates:", df.shape)

# -------------------------------
# STEP 3: Clean 'rating' column
# Issues:
# - '--' represents missing rating
# -------------------------------
df['rating'] = df['rating'].replace('--', None)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# -------------------------------
# STEP 4: Clean 'rating_count' column
# Issues:
# - 'Too Few Ratings' is non-numeric
# -------------------------------
df['rating_count'] = df['rating_count'].replace('Too Few Ratings', None)
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

# -------------------------------
# STEP 5: Clean 'cost' column
# Issues:
# - Contains currency symbols and encoding characters
# -------------------------------
df['cost'] = (
    df['cost']
    .astype(str)
    .str.replace(r'[^\d]', '', regex=True)
)

df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

# -------------------------------
# STEP 6: Drop irrelevant columns
# These are not needed for modeling
# -------------------------------
columns_to_drop = ['lic_no']
for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# -------------------------------
# STEP 7: Handle missing values
# Numerical → median
# Categorical → drop rows
# -------------------------------
df['rating'].fillna(df['rating'].median(), inplace=True)
df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
df['cost'].fillna(df['cost'].median(), inplace=True)

df.dropna(subset=['city', 'cuisine'], inplace=True)

# -------------------------------
# STEP 8: Final sanity check
# -------------------------------
print("\nFinal dataset info:")
print(df.info())

# -------------------------------
# STEP 9: Save cleaned dataset
# -------------------------------
df.to_csv("data/cleaned_data.csv", index=False)

print("\n✅ Cleaned data saved as: data/cleaned_data.csv")
