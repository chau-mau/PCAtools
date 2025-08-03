import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
file_path = "your_file.xlsx"  # Replace if needed
df = pd.read_excel(file_path)

# Ensure columns are strings
df.columns = df.columns.astype(str)

# Define fixed effect columns
fixed1 = df.columns[1]  # Column 2 (for shape binning)
fixed2 = df.columns[2]  # Column 3 (for color)

# Drop rows with any missing values
df_clean = df.dropna(how='any').copy()

# Convert fixed1 to numeric safely
df_clean[fixed1] = pd.to_numeric(df_clean[fixed1], errors='coerce')
df_clean = df_clean.dropna(subset=[fixed1])

# Bin fixed1 into shape categories
bins = [0, 3, 6, 9, 12, 15]
labels = ['0-3', '3-6', '6-9', '9-12', '12-15']
df_clean['ShapeBin'] = pd.cut(df_clean[fixed1], bins=bins, labels=labels, include_lowest=True)

# Prepare color values from fixed2
df_clean[fixed2] = pd.to_numeric(df_clean[fixed2], errors='coerce')
df_clean = df_clean.dropna(subset=[fixed2])
color_column = df_clean[fixed2]
color_map = {1: 'darkblue', 2: 'deeppink'}
sex_label = {1: 'Male', 2: 'Female'}

# Select only numeric columns for PCA
numeric_df = df_clean.select_dtypes(include='number')
X = StandardScaler().fit_transform(numeric_df)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# PCA DataFrame for plotting
pca_df = pd.DataFrame({
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'Color': color_column.loc[df_clean.index],
    'ShapeBin': df_clean['ShapeBin']
})

# Shape marker mapping
shape_markers = ['o', 's', '^', 'D', 'X']
shape_map = dict(zip(labels, shape_markers))

# ====================
# 📊 PCA Scatter Plot
# ====================
plt.figure(figsize=(10, 8))
for shape_bin in labels:
    for color_val in [1, 2]:
        subset = pca_df[(pca_df['ShapeBin'] == shape_bin) & (pca_df['Color'] == color_val)]
        if not subset.empty:
            plt.scatter(
                subset['PC1'], subset['PC2'],
                c=color_map[color_val],
                marker=shape_map[shape_bin],
                label=f"{shape_bin}, {sex_label[color_val]}",
                edgecolor='black',
                alpha=0.8
            )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.title("PCA Scatter Plot")
plt.legend(title=f"{fixed1} bin & {fixed2}", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_plot_shapes_colors.png", dpi=600)
plt.show()

# ================
# 📈 Scree Plot
# ================
plt.figure(figsize=(8, 6))
explained_var = pca.explained_variance_ratio_ * 100
sns.barplot(x=[f'PC{i+1}' for i in range(len(explained_var))], y=explained_var, color='skyblue')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("scree_plot.png", dpi=600)
plt.show()

# =========================
# 🔥 Trait Loadings Heatmap
# =========================
loadings = pd.DataFrame(pca.components_[:2, :], columns=numeric_df.columns, index=['PC1', 'PC2'])

plt.figure(figsize=(10, 4))
sns.heatmap(loadings.T, annot=True, cmap='vlag', center=0)
plt.title("Trait Contributions to PC1 and PC2")
plt.tight_layout()
plt.savefig("pca_trait_heatmap.png", dpi=600)
plt.show()
