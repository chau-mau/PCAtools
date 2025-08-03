import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read Excel data
df = pd.read_excel('your_file.xlsx')

# Drop some column from features
features_df = df.drop(columns=[df.columns[0], df.columns[1], df.columns[2]])

# Replace empty strings with NaN and convert numeric columns
features_df = features_df.apply(pd.to_numeric, errors='coerce')

# Extract and encode groups for color and shape
group_col = df.columns[1]  # typically 'Sex'
age_col = df.columns[2]    # typically 'Age'

# Handle grouping for color
color_group = df[group_col].map({'c1': 'Male', 'c2': 'Female'})

# Categorize Age into bins for shape coding
shape_bins = pd.cut(df[age_col], bins=[0, 3, 6, 9, 12, 15], labels=['0-3', '3-6', '6-9', '9-12', '12-15'])

# Drop rows with missing values
valid_idx = features_df.dropna().index
X = features_df.loc[valid_idx]
color_group = color_group.loc[valid_idx]
shape_bins = shape_bins.loc[valid_idx]

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Group'] = color_group.values
pca_df['AgeGroup'] = shape_bins.values

# Save PCA coordinates
pca_df.to_csv('PCA_output.csv', index=False)

# Scree plot
plt.figure(figsize=(8, 6), dpi=600)
explained_var = pca.explained_variance_ratio_ * 100
plt.bar(['PC1', 'PC2'], explained_var, color='steelblue')
plt.ylabel('Variance Explained (%)')
plt.title('Scree Plot')
plt.tight_layout()
plt.savefig('scree_plot.png', dpi=600)
plt.close()

# PCA scatter plot with shapes
plt.figure(figsize=(10, 8), dpi=600)
shapes = {'0-3': 'o', '3-6': 's', '6-9': '^', '9-12': 'D', '12-15': 'P'}
for shape_group in shapes:
    for color in ['Male', 'Female']:
        subset = pca_df[(pca_df['AgeGroup'] == shape_group) & (pca_df['Group'] == color)]
        plt.scatter(subset['PC1'], subset['PC2'],
                    label=f'{color}, Age {shape_group}',
                    marker=shapes[shape_group], s=100, alpha=0.7)

plt.xlabel(f"PC1 ({explained_var[0]:.2f}%)")
plt.ylabel(f"PC2 ({explained_var[1]:.2f}%)")
plt.title('PCA Scatter Plot')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_plot.png', dpi=600)
plt.close()

# Heatmap of loading scores
loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=X.columns)

plt.figure(figsize=(8, 10), dpi=600)
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('Loading Scores Heatmap')
plt.tight_layout()
plt.savefig('loadings_heatmap.png', dpi=600)
plt.close()

# Save loadings to CSV
loadings.to_csv('pca_loadings.csv')

