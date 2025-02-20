import pandas as pd
from scipy.stats import ttest_ind

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

original_file = 'complete_dataset.xlsx'
generated_file = 'generated_100.xlsx'

df_original = pd.read_excel(original_file)
df_generated = pd.read_excel(generated_file)

original_desc = df_original[demographic_columns].describe().T
generated_desc = df_generated[demographic_columns].describe().T

original_desc = original_desc.rename(columns=lambda x: f"Original {x}")
generated_desc = generated_desc.rename(columns=lambda x: f"Generated {x}")

combined_desc = original_desc.join(generated_desc)

ttest_results = {}
for col in demographic_columns:
    t_stat, p_val = ttest_ind(df_original[col], df_generated[col], equal_var=False)
    ttest_results[col] = {"t-statistic": t_stat, "p-value": p_val}
ttest_results_df = pd.DataFrame(ttest_results).T

output_file = 'comparison_summary.xlsx'
with pd.ExcelWriter(output_file) as writer:
    combined_desc.to_excel(writer, sheet_name='Descriptive Stats')
    ttest_results_df.to_excel(writer, sheet_name='T-Test Results')

print(f"Summary comparisons saved to '{output_file}'.")
