import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

non_demographic_columns = [
    "No. ",
    "Category",
    "State/Market Region",
    "Location",
    "Reach \nPer Week",
    "Screen Size\n(H)  x (W) ",
    "Material Size\n(H)  x (W) ",
    "Operation Hour/Day",
    "Material/File Format",
    "Ad Duration",
    "No. of Exposure/Daily (Min)",
    "No. of Exposure/\nMonthly (Min)",
    "Max. Ad Slot",
    "Loops",
    "No. of Screens",
    "Location GPS Coordinates",
    "Latitude",
    "Longitude",
    "Gross Rate Card (RM)\n(Per Month)",
    "Gross Special Rate Card \n(RM) (Per Month)",
    "Content Management Fee (RM)",
    "CPM",
    "Board Viewing Distance",
    "Angle of view",
    "Min Exposures Per Day",
    "Reach/Eyeballs (Daily)",
    "Reach/Eyeballs (Monthly)",
    "Traffic/Vehicles (Daily)",
    "Traffic/Vehicles (Monthly)",
    "Dwell Time",
    "Location Type 1",
    "Location Type 2",
    "Orientation",
    "Device Count"
]

demographic_columns = [
    "Games - Sports",
    "Social & Casual Gamers",
    "Foodies/Cooking Enthusiasts",
    "Restaurant Frequenters",
    "Age : 18-24",
    "Age : 25-34",
    "Age : 35-44",
    "Age : 45-54",
    "Age : 55 and above",
    "Females",
    "Males",
    "Families with Kids",
    "High Income",
    "Avg Income",
    "Low Income",
    "Young Mothers",
    "Fitness Enthusiasts",
    "Gym Goers",
    "Gamers",
    "Movie Goers",
    "Tech Enthusiasts",
    "College Students",
    "Singles",
    "Beauty and Health buyers",
    "Big Box Retailer Shoppers",
    "Consumer Electronics Buyers",
    "Shoppers/Fashion Followers",
    "Consumer packaged goods (CPG)",
    "Restaurant Customers (QSR)",
    "Toy's Buyers",
    "Young & Hip Shoppers",
    "Pharmacy Patrons",
    "Upscale Shoppers",
    "Sports & Outdoors Buyers",
    "Golf Enthusiasts",
    "Sports Enthusiasts",
    "Mobile Device: Apple",
    "Air Travelers",
    "Business Travelers",
    "Public transport users",
    "Leisure Travelers",
    "Android users"
]

data = pd.read_excel('complete_dataset.xlsx')

demo_data = data[demographic_columns]

print("Training Demographic Data Sample:")
print(demo_data.head())

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(demo_data)

num_samples_503 = 503
samples_503, _ = gmm.sample(num_samples_503)
df_503 = pd.DataFrame(samples_503, columns=demographic_columns)


for col in demographic_columns:
    lower_bound = demo_data[col].min()  # minimum value from training data for lower bound
    df_503[col] = df_503[col].round().clip(lower=lower_bound).astype(int)

for col in non_demographic_columns:
    df_503[col] = np.nan

print("\n 503 Generated Rows (Generative Model)")
print(df_503.head())

df_503.to_excel('generated_503.xlsx', index=False)
print("\n503 generated rows saved to 'generated_503.xlsx'.")

gmm_503 = GaussianMixture(n_components=3, random_state=42)
gmm_503.fit(df_503[demographic_columns])

num_samples_100 = 100
samples_100, _ = gmm_503.sample(num_samples_100)
df_100 = pd.DataFrame(samples_100, columns=demographic_columns)

for col in demographic_columns:
    lower_bound = demo_data[col].min()
    df_100[col] = df_100[col].round().clip(lower=lower_bound).astype(int)

for col in non_demographic_columns:
    df_100[col] = np.nan

print("\n100 Generated Rows (Second Generation)")
print(df_100.head())

df_100.to_excel('generated_100.xlsx', index=False)
print("\n100 generated rows saved to 'generated_100.xlsx'.")

print("\nOriginal Demographic Data Statistics:")
print(demo_data.describe())

print("\n503-Row Generated Demographic Data Statistics:")
print(df_503[demographic_columns].describe())

print("\n100-Row Second Generation Demographic Data Statistics:")
print(df_100[demographic_columns].describe())

