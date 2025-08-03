# PCAtools
  
This repository contains two Python scripts for performing Principal Component Analysis (PCA) on biometric trait datasets using Excel input. These tools are tailored for biological datasets where traits like body measurements are analyzed to detect patterns of variation.

---

## 📂 Contents

1. `pca_with_shape_and_color.py`  
   - Performs PCA with grouping based on two fixed effects (e.g., age bin and sex).
   - Plots:
     - PCA scatter plot with color and shape coding
     - Scree plot showing explained variance
     - Trait loading heatmap
   - Suitable when both group and age variables are numeric or encodable.

2. `pca_cleaned_columns.py`  
   - Removes non-numeric metadata (e.g., Animal ID, Sex, Age) before performing PCA.
   - Encodes 'Sex' and binned 'Age' for color and shape respectively.
   - Generates PCA coordinates, scree plot, and heatmap of trait loadings.
