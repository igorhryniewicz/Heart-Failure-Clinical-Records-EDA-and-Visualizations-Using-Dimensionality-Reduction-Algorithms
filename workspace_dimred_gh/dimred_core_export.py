# %% [markdown]
# ___
# # Loading the data and libraries
# ___

# %%
%load_ext autoreload
%autoreload 2

import os
import sys
import warnings

while any(marker in os.getcwd() for marker in ['workspace_dimred_gh']):
    os.chdir("..")

sys.path.append('classes_and_functions_dimred_gh')

current_directory = os.getcwd()
current_directory

# %%
# Importing external packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import statsmodels.api as sm
import umap.umap_ as umap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from statsmodels.formula.api import ols
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

# Importing custom-made functions and classes
from classes_and_functions_dimred_gh.custom_functions_classes_dimred import (
    unique_column_content_check,
    corr_matrix_dataframe,
    make_mi_scores,
    plot_mi_scores,
    skewness,
    DropColumnTransformer,
    CustomOutlierRemoverInterquartile,
    CustomMinMaxScaler,
    plot_indices_relation,
    plot_violin_features,
    plot_violin_with_binary_hue,
    plot_histograms_nonbinary,
    plot_histograms_nonbinary_logarithmic,
    plot_pairplots_kde_hue,
    plot_class_distribution,
    plot_algo3d,
    plot_explained_variance,
    show_pca_weights,
    lda_transform_plot,
)

# %%
raw_data = pd.read_csv('attachments_dimred_gh/heart_failure_clinical_records_dataset.csv')
raw_data.head()

# %% [markdown]
# ___
# # ***EDA*** - before cleaning
# ___

# %% [markdown]
# ### *basic* ***EDA***

# %%
raw_data.shape

# %%
raw_data.dtypes

# %% [markdown]
# There are no categorical features. *age* is a float and should be changed. 

# %%
raw_data.isnull().sum()

# %% [markdown]
# Fortunately there are no null values in this dataset.

# %%
plot_indices_relation(raw_data, (12, 15))

# %% [markdown]
# There is a correlation between indices and *time*, which may be explained by the fact that this dataset is sorted by *time* feature.

# %% [markdown]
# #### Distribution

# %%
print("\nSummary statistics for numerical variables:")
raw_data.describe()

# %% [markdown]
# We have some small conclusions:
# - *creatinine_phosphokinase* may be right-skewed as **mean** and **median** differ. Also there seem to be outliers as the **maximum** is a lot higher than the **mean**,
# - *platelets* and *serum_creatinine* definitely have outliers since their **maximum** values are extremaly high,
# - There is an imbalance in the target feature (*DEATH_EVENT*), as $32\%$ of people died of heart failure.

# %%
skewness(raw_data)

# %% [markdown]
# As hypothesised *creatinine_phosphokinase* is highly right-skewed, as are *serum_creatinine* and *platelets*. Most of the features have midly right-skewed distribution.

# %%
binary_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
raw_data_nonbinary = raw_data.drop(binary_columns, axis=1)

# %%
plot_violin_features(raw_data_nonbinary, (18, 12))

# %% [markdown]
# Violin plots confirm the existence of outliers in several fetures (they cause longer tails of the distributions).

# %%
plot_violin_with_binary_hue(raw_data, binary_columns, (25, 6))

# %% [markdown]
# After checking violin plots with distinction for different binary features we can draw some conclusions:
# - for now features with visible outliers are not informative,
# - *time* with respect to *DEATH_EVENT* differ greatly in **mean** and distribution,
# - *ejection_fraction* with respect to *DEATH_EVENT* differ visibly in **mean**.

# %%
plot_histograms_nonbinary(raw_data_nonbinary, (20, 8))

# %% [markdown]
# We can suspect some normality of the data, except for *time* which seems uniformly distributed.

# %%
plot_histograms_nonbinary_logarithmic(raw_data_nonbinary, ['creatinine_phosphokinase', 'serum_creatinine'], (10, 4))

# %% [markdown]
# We used the function $\log (1+x)$ on *creatinine_phosphokinase* and *serum_creatinine* so see whether the outliers break normality of those features. We can conlude that *creatinine_phosphokinase* is definitely not normally distributed.

# %%
for key, value in unique_column_content_check(raw_data_nonbinary).items():
    print(f'Column name: {key}\nUnique values:\n{value[0]}\nTotal unique values:{value[1]}\n\n')


# %% [markdown]
# We found a reason for why *platelets* and *age* are a float type - we can safely round the values to then change this feature to int.
# 
# Also we can clearly read and find the outliers of previously mentioned features.

# %%
plot_pairplots_kde_hue(raw_data, binary_columns)

# %%
correlation_matrix = raw_data_nonbinary.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, linewidth=.5, fmt='.2f', cmap='crest')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
corr_matrix_dataframe(raw_data_nonbinary)

# %% [markdown]
# We can't find any correlation between features that would be of our interest.

# %%
mi_scores = make_mi_scores(raw_data.drop('DEATH_EVENT', axis=1), raw_data['DEATH_EVENT'])

plot_mi_scores(mi_scores)

# %% [markdown]
# We find that before cleaning features above *creatinine_phosphokinase* have some significance with respect to *DEATH_EVENT*. 
# 
# *time* is greatly in the lead.

# %% [markdown]
# #### Class label distribution
# This dataset is **imbalanced** with almost $2\times$ samples labeled $0$.

# %%
plot_class_distribution(raw_data['DEATH_EVENT'], 'DEATH_EVENT')

# %% [markdown]
# We can see the confirmation of the fact that the data is imbalanced for our target.

# %% [markdown]
# ___
# # ***Cleaning***
# ___

# %%
raw_data['age'] = raw_data['age'].round().astype('int64')
raw_data['platelets'] = raw_data['platelets'].round().astype('int64')

# %%
raw_data.dtypes


# %% [markdown]
# We changed the *age* and *platelets* to int.

# %%
data_cleaning = make_pipeline(
    FunctionTransformer(lambda X: X.drop_duplicates(), validate=False),
    CustomOutlierRemoverInterquartile(factor=2.5),
)
df_cleaned = data_cleaning.fit_transform(raw_data)
df_cleaned.head()

# %% [markdown]
# We created a custom pipeline, which removes the duplicates, and gets rid of the outliers based on IQR, because most of the distributions were not normal, so using z-score would not be beneficial.

# %% [markdown]
# ___
# # ***EDA*** - after cleaning
# ___

# %% [markdown]
# ### *basic* ***EDA***

# %%
df_cleaned.shape

# %% [markdown]
# Number dropped from $299\to 251$ after cleaning.

# %%
df_cleaned.dtypes

# %%
plot_indices_relation(df_cleaned, (12, 15))

# %%
print("\nSummary statistics:")
df_cleaned.describe()

# %% [markdown]
# We can see we got rid of previously prominent outliers.

# %%
warnings.filterwarnings(action='ignore')
df_cleaned_nonbinary = df_cleaned.drop(binary_columns, axis=1)

plot_violin_features(df_cleaned_nonbinary, (18, 12))

# %% [markdown]
# We can see the tails got smaller, but without loss of so much high/low valued data.

# %%
warnings.filterwarnings(action='ignore')
plot_violin_with_binary_hue(df_cleaned, binary_columns, (25, 6))

# %% [markdown]
# (*time* $\lor$ *serum_creatinine* $\lor$ *age*) with respect to *DEATH_EVENT* has significant difference in **mean**.

# %%
plot_histograms_nonbinary(df_cleaned_nonbinary, (20, 8))

# %% [markdown]
# Our previous hypotheses about normality of the features are to be probably rejected.

# %%
plot_histograms_nonbinary_logarithmic(df_cleaned_nonbinary, ['creatinine_phosphokinase', 'serum_creatinine'], (10, 4))

# %% [markdown]
# Again we used $\log (1+x)$ to see some results, but it did not give us anything significant.

# %%
for key, value in unique_column_content_check(df_cleaned_nonbinary).items():
    print(f'Column name: {key}\nUnique values:\n{value[0]}\nTotal unique values:{value[1]}\n\n')

# %%
plot_pairplots_kde_hue(df_cleaned, binary_columns)

# %%
correlation_matrix = df_cleaned_nonbinary.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, linewidth=.5, fmt='.2f', cmap='crest')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
corr_matrix_dataframe(df_cleaned_nonbinary)

# %%
mi_scores = make_mi_scores(df_cleaned.drop('DEATH_EVENT', axis=1), df_cleaned['DEATH_EVENT'])

plot_mi_scores(mi_scores)

# %% [markdown]
# We do not see prominent change in ***MI*** scores.
# 
# Now only features above *sex* are of any significance.

# %% [markdown]
# ### ***EDA*** - statistical tests (for significance level $\alpha = 0.05$)

# %%
binary_permutations = list(itertools.combinations(binary_columns, 2))
chi2_independency_array = []

summed_table = df_cleaned.shape[0]
for columns in binary_permutations:
    cross_table = pd.crosstab(df_cleaned[columns[0]], df_cleaned[columns[1]], rownames=[columns[0]], colnames=[columns[1]])
    chi2, p, dof, expected = stats.chi2_contingency(cross_table)
    if p <= 0.05:
        phi_coefficient = np.sqrt(chi2 / summed_table)
    else:
        phi_coefficient = np.nan

    chi2_independency_array.append([columns[0], columns[1], round(p, 4), phi_coefficient])

chi2_independency_df = pd.DataFrame(data=chi2_independency_array, columns=['binary_1', 'binary_2', 'chi2_p_value', 'phi_coefficient'])

chi2_independency_df.sort_values(by='chi2_p_value', ascending=False)

# %%
chi2_independency_df[chi2_independency_df['chi2_p_value'] <= 0.05].sort_values(by='chi2_p_value', ascending=False)

# %% [markdown]
# After performing $\chi^2$ independence test of binary data, we concluded that *diabetes* and *sex*, *smoking* and *sex* are dependent on each other, with moderate strengh in (*smoking* | *sex*)

# %%
plt.figure(figsize=(20, 6))
plt.subplots_adjust(left=0, bottom=0, right=0.8, top=1, wspace=0.3, hspace=0.4)

for i, column in enumerate(raw_data_nonbinary.columns):
    plt.subplot(2, 4, i+1)
    stats.probplot(df_cleaned[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot | {column}')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
plt.show()

# %%
shapiro_wilk_array = []
for column in raw_data_nonbinary.columns:
    sw_stat, sw_p = stats.shapiro(df_cleaned[column])
    shapiro_wilk_array.append([column, round(sw_p, 4)])

shapiro_wilk_df = pd.DataFrame(data=shapiro_wilk_array, columns=['feature', 'sw_p_value'])

shapiro_wilk_df

# %% [markdown]
# After perfoming ***Shapiro-Wilk*** test on normality, we conclude non of the features have normal distribution. It can be somewhat seen on Q-Q plots. 

# %%
ANOVA_array = []

for column in raw_data_nonbinary.columns:
    for bi_column in binary_columns:
        formula = f'{column} ~ C({bi_column})'
        model = ols(formula, data=df_cleaned).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        ANOVA_array.append([f'{column}',  f'{bi_column}', round(anova_table.iloc[0, -1], 4)])
ANOVA_df = pd.DataFrame(data = ANOVA_array, columns=['Feature', 'Distinction', 'p_value'])

ANOVA_df['p_value'] = ANOVA_df['p_value'].astype('float64')

ANOVA_df.sort_values(by='p_value', ascending=False)

# %% [markdown]
# We can see that even though *creatinine_phosphokinase* has non-zero ***MI*** score, it has approximately the same mean when conditioned on *DEATH_EVENT*.

# %%
ANOVA_df[ANOVA_df['p_value'] <= 0.05].sort_values(by='p_value', ascending=False)

# %% [markdown]
# After perfoming ***ANOVA*** on each non-binary feature with respect to a binary one, we can see it coincides with our outcome of mutual information.

# %% [markdown]
# #### Experimental ***EDA***

# %% [markdown]
# We attempt to categorise numerical features, in order to perform independence tests on them.

# %%
df_experimental = df_cleaned.copy().drop(binary_columns[:-1], axis=1)
df_experimental_categorical = df_experimental.copy() 

# %%
df_experimental.describe()

# %%
for key, value in unique_column_content_check(df_experimental).items():
    print(f'Column name: {key}\nUnique values:\n{value[0]}\nTotal unique values:{value[1]}\n\n')

# %%
df_experimental_categorical['age'] = df_experimental['age'].apply(lambda x: 'low' if x <= 58 else ('medium' if x <= 76 else 'high'))
df_experimental_categorical['creatinine_phosphokinase'] = df_experimental['creatinine_phosphokinase'].apply(lambda x: 'normal' if x <= 132 else ('high' if x < 700 else 'extremely high'))
df_experimental_categorical['ejection_fraction'] = df_experimental['ejection_fraction'].apply(lambda x: 'low' if x <= 40 else ('medium' if x <= 50 else 'normal'))
df_experimental_categorical['platelets'] = df_experimental['platelets'].apply(lambda x: 'low' if x <= 150_000 else ('normal' if x <= 400_000 else 'high'))
df_experimental_categorical['serum_creatinine'] = df_experimental['serum_creatinine'].apply(lambda x: 'low' if x <= 0.75 else ('normal' if x <= 1.3 else 'high'))
df_experimental_categorical['serum_sodium'] = df_experimental['serum_sodium'].apply(lambda x: 'low' if x <= 134 else ('normal' if x <= 145 else 'high'))
df_experimental_categorical['time'] = df_experimental['time'].apply(lambda x: 'high prob' if x <= 75 else ('moderate prob' if x <= 150 else 'low prob'))


df_experimental_categorical

# %% [markdown]
# Our new labels:
# - *age* || $x \leqslant 58\mapsto$ **low** | $58 < x \leqslant 76\mapsto$ **medium** | $76 < x\mapsto$ **high**
# 
# - *creatinine_phosphokinase* || $x \leqslant 132\mapsto$ **normal** | $132 < x \leqslant 700\mapsto$ **high** | $700 < x\mapsto$ **extremely high**
# - *ejection_fraction* || $x \leqslant 40\mapsto$ **low** | $40 < x \leqslant 50\mapsto$ **medium** | $50 < x\mapsto$ **normal**
# - *platelets* || $x \leqslant 150000\mapsto$ **low** | $150000 < x \leqslant 400000\mapsto$ **normal** | $400000 < x\mapsto$ **high**
# - *serum_creatinine* || $x \leqslant 0.75\mapsto$ **low** | $0.75 < x \leqslant 1.3\mapsto$ **normal** | $1.3 < x\mapsto$ **high**
# - *serum_sodium* || $x \leqslant 134\mapsto$ **low** | $134 < x \leqslant 145\mapsto$ **normal** | $145 < x\mapsto$ **high**
# - *time* || $x \leqslant 75\mapsto$ **low** | $75 < x \leqslant 150\mapsto$ **normal** | $150 < x\mapsto$ **high**
# 
# Everything is inspired by medical sources and/or our analysis of the data. Some are approximated due to differences in threshold due to gender.

# %%
ordinal_permutations = list(itertools.combinations(df_experimental_categorical.columns, 2))
chi2_independency_array = []

summed_table = df_experimental_categorical.shape[0]
for columns in ordinal_permutations:
    cross_table = pd.crosstab(df_experimental_categorical[columns[0]], df_experimental_categorical[columns[1]], rownames=[columns[0]], colnames=[columns[1]])
    chi2, p, dof, expected = stats.chi2_contingency(cross_table)
    if p <= 0.05:
        cramers_v = np.sqrt(chi2 / (summed_table * (min(cross_table.shape) - 1)))
    else:
        cramers_v = np.nan

    chi2_independency_array.append([columns[0], columns[1], round(p, 4), cramers_v])

chi2_independency_df_ordinal = pd.DataFrame(data=chi2_independency_array, columns=['ordinal 1', 'ordinal 2', 'chi2 p_value', 'cramers V'])

chi2_independency_df_ordinal.sort_values(by='chi2 p_value', ascending=False)

# %%
chi2_independency_df_ordinal[chi2_independency_df_ordinal['chi2 p_value'] <= 0.05].sort_values(by='chi2 p_value', ascending=False)

# %% [markdown]
# Having in mind ***Violin Plots***, ***Mutual Information***, and ***ANOVA*** we can *clearly* see the prominence of *time*, *serum_creatinine*, *age*, *serum_sodium*, and *ejection_fraction* in relation to *DEATH_EVENT*.

# %% [markdown]
# # ***Preprocessing***

# %%
preprocessing_pipeline = make_pipeline(
    CustomMinMaxScaler(columns=['age', 'ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time']),
    DropColumnTransformer(columns=['smoking', 'anaemia', 'platelets', 'creatinine_phosphokinase', 'diabetes', 'high_blood_pressure', 'sex']),
)
df_preprocessed_with_target = preprocessing_pipeline.fit_transform(df_cleaned)
df_preprocessed_with_target

# %% [markdown]
# ## Resolving Imbalance
# As shown in EDA section, one class dominates the other one by covering approximately 68% of the instances. This causes an imbalance in the dataset, which is resolved below.
# 
# We can use: 
# - Over-sampling: create new synthetic examples in the minority class (with SMOTE)
# - Under-sampling: delete or merge examples in the majority class (with Random Sampler)
# - A combo of both (with SMOTEENN)

# %%
### Oversample with SMOTE
df_copy = df_preprocessed_with_target.copy()

# Split features and target
y = df_copy['DEATH_EVENT']
X = df_copy.drop(['DEATH_EVENT'], axis=1)

smote_sampler = SMOTE(random_state=42)
oversampled_X, oversampled_y = smote_sampler.fit_resample(X, y)

# Plot resulting distribution
plot_class_distribution(oversampled_y, 'DEATH_EVENT')


### Undersample with Randome Under Sampler
rand_sampler = RandomUnderSampler(random_state=42)
undersampled_X, undersampled_y = rand_sampler.fit_resample(X, y)

# Plot resulting distribution
plot_class_distribution(undersampled_y, 'DEATH_EVENT')


### SMOTEENN
smoteenn_sampler = SMOTEENN(random_state=42)
combo_sampled_X, combo_sampled_y = smoteenn_sampler.fit_resample(X, y)

# Plot resulting distribution
plot_class_distribution(combo_sampled_y, 'DEATH_EVENT')


print(f'Original shape: {X.shape}')
print(f'Oversampled dataset shape:" {oversampled_X.shape}')
print(f'Undersampled dataset shape: {undersampled_X.shape}')
print(f'Sampled dataset shape (SMOTEENN): {combo_sampled_X.shape}')



# %% [markdown]
# ### ***LDA*** (Linear Discriminant Analysis)
# Since LDA is a supervised DR algorithm, we need to include the target in the dataset. For this purpose, we'll be employing our previously resampled data.
# 
# We will have only one component because the number of components (aka linear discriminants) is determined by the number of unique labels in the target variable; `DEATH_EVENT` , i.e., our target, is a binary variable.

# %% [markdown]
# Linear Discriminant Analysis (LDA) is a technique used to find the best separation between different groups or categories in a dataset. Unlike PCA, which focuses on capturing the most variance in the data, LDA aims to maximize the separation between different classes while minimizing the variation within each class.
# 
# Mathematically, LDA works by:
# 
# 1. Calculating the Means: It computes the mean for each class and the overall mean of the data.
# 2. Within-Class and Between-Class Variance: It then calculates the within-class scatter (how much the data points in each class vary) and the between-class scatter (how much the means of the different classes vary from each other).
# 3. Maximizing Separation: LDA finds the linear combinations of the original variables that maximize the ratio of between-class variance to within-class variance. These combinations are called linear discriminants.
# 
# The result is a transformation that projects the data onto a new space where the classes are as distinct as possible. This is particularly useful for classification tasks, where the goal is to distinguish between different categories based on their features.

# %%
# Using imbalanced data
title = 'LDA with Imbalanced Data'
lda_transform_plot(X, y, title=title)

# Using oversampled data
title = 'LDA with Oversampled Data'
lda_transform_plot(oversampled_X, oversampled_y, title=title)

# Using undersampled data
title = 'LDA with Unversampled Data'
lda_transform_plot(undersampled_X, undersampled_y, title)

# Using data sampled with SMOTEENN
title = 'LDA with Sampled Data using SMOTEENN'
lda_transform_plot(combo_sampled_X, combo_sampled_y, title)

# %% [markdown]
# #### Interpretation - LDA
# Each point represents a data point (instance).
# As can be seen above, the most discriminative performance is demonstrated in the fourth plot. The instances seem to be perfectly separated along the x-axis (LD1).
# 
# Based on this outcome, we will continue our journey using ***SMOTEENN*** sampling for imbalance.

# %% [markdown]
# ### PCA for preprocessed data **without** sampling

# %% [markdown]
# PCA works by mathematically transforming the data into a new coordinate system. It starts by calculating the covariance matrix of the data, which captures how much the variables vary together. Then, it finds the eigenvectors and eigenvalues of this matrix.
# 
# The eigenvectors represent the directions of maximum variance (principal components), and the eigenvalues indicate the amount of variance along these directions. The data is then projected onto these principal components, resulting in a new set of uncorrelated variables that capture the most significant patterns in the original dataset. The principal components are ordered by the amount of variance they explain, allowing for dimensionality reduction by selecting only the top components.

# %%
# Drop target to use data in PCA, etc.
df_preprocessed = df_preprocessed_with_target.drop(['DEATH_EVENT'], axis=1)

# %%
# PCA for 3 components
pca_3d = PCA(n_components=3)
df_pca_3d = pca_3d.fit_transform(df_preprocessed)

plot_algo3d(df_pca_3d, df_cleaned['DEATH_EVENT'], algo_name="PCA", target_name='DEATH_EVENT')

# %% [markdown]
# After taking a glance at scatter plots of all the 3 principal components of **PCA** one can suspect that ***PC1*** and ***PC3*** convey the most information about *DEATH_EVENT*.

# %%
plot_explained_variance(pca_3d)

# %% [markdown]
# We managed to get almost $80\%$ of variance explained in our ***3 PCs*** using 5 features for it. 

# %%
show_pca_weights(df_pca_3d, pca_3d, df_preprocessed)

# %% [markdown]
# - ***PC1***: *age*, *serum_creatinine* and *time* have the highest weight
# 
# - ***PC2*** and ***PC3***: there are no near zero weights, so we cannot try to exclude any features

# %%
component_names_pca_3d = [f"PC{i+1}" for i in range(df_pca_3d.shape[1])]
df_pca_3d_structurized = pd.DataFrame(df_pca_3d, columns=component_names_pca_3d)

mi_scores_pca = make_mi_scores(df_pca_3d_structurized, df_cleaned['DEATH_EVENT'])

plot_mi_scores(mi_scores_pca)

# %% [markdown]
# We can see the confirmation of our guess. ***PC1*** conveys the most information, with ***PC2*** and ***PC3*** being close in value, but low. 
# 
# It explains what we saw on our plots - ***PC2*** and ***PC3*** divide people terribly, whilst (***PC1*** and ***PC2***) or (***PC1*** and ***PC3***) do better because of ***PC1*** [***PC3*** performs better visually, since it has slightly higher ***MI*** score]. 

# %% [markdown]
# ### PCA for preprocessed data **with** ***SMOTEENN*** sampling

# %%
# PCA for 3 components
pca_3d_smoteenn = PCA(n_components=3)
df_pca_3d_smoteenn = pca_3d_smoteenn.fit_transform(combo_sampled_X)

plot_algo3d(df_pca_3d_smoteenn, combo_sampled_y, algo_name="PCA", target_name='DEATH_EVENT')

# %%
plot_explained_variance(pca_3d_smoteenn)

# %% [markdown]
# Explained Variance jumped from $80\%\to 83\%$.

# %%
show_pca_weights(df_pca_3d_smoteenn, pca_3d_smoteenn, df_preprocessed)

# %%
component_names_pca_3d_smoteenn = [f"PC{i+1}" for i in range(df_pca_3d_smoteenn.shape[1])]
df_pca_3d_smoteenn_structurized = pd.DataFrame(df_pca_3d_smoteenn, columns=component_names_pca_3d_smoteenn)

mi_scores_pca_smoteenn = make_mi_scores(df_pca_3d_smoteenn_structurized, combo_sampled_y)

plot_mi_scores(mi_scores_pca_smoteenn)

# %% [markdown]
# We can see that ***MI*** score for ***PC3*** is almost $0$, so we can confidently convey all of the information using only ***PC1*** and ***PC2***.

# %% [markdown]
# ### ***UMAP***

# %% [markdown]
# Uniform Manifold Approximation and Projection (UMAP) is a technique used to reduce the dimensionality of data, making it easier to visualize and analyze. Like PCA, UMAP simplifies complex datasets, but it focuses more on preserving the local structure of the data rather than just capturing the most variance.
# 
# Mathematically, UMAP works by:
# 
# 1. Constructing a Graph: It first constructs a graph representing the data points, where each point is connected to its nearest neighbors, forming a high-dimensional space.
# 2. High-Dimensional Representation: UMAP uses a mathematical framework called "fuzzy simplicial sets" to understand the local structure of this space.
# 3. Optimization: It then optimizes the layout by minimizing a cost function that measures how well the low-dimensional representation preserves the high-dimensional structure.
# 
# The result is a low-dimensional representation, often in 2D or 3D, where similar data points are placed close together and dissimilar ones are pushed apart. This makes UMAP particularly useful for visualizing complex data and uncovering patterns, clusters, or relationships that might not be apparent in the high-dimensional space.

# %%
reducer = umap.UMAP(n_components=3, random_state=3721)
embedding = reducer.fit_transform(df_preprocessed)

plot_algo3d(embedding, df_preprocessed_with_target['DEATH_EVENT'], algo_name='UMAP', target_name="DEATH_EVENT")

# %%
component_names_umap_3d = [f"UMAP{i+1}" for i in range(embedding.shape[1])]
df_umap_3d_structurized = pd.DataFrame(embedding, columns=component_names_umap_3d)

mi_scores_umap = make_mi_scores(df_umap_3d_structurized, df_preprocessed_with_target['DEATH_EVENT'])

plot_mi_scores(mi_scores_umap)

# %% [markdown]
# We can see that ***UMAP2*** conveys the most (but still not much) information about the target, ***UMAP1*** and ***UMAP2*** provides small amount of information. 

# %%
reducer_smoteenn = umap.UMAP(n_components=3, random_state=3721)
embedding_smoteenn = reducer_smoteenn.fit_transform(combo_sampled_X)

plot_algo3d(embedding_smoteenn, combo_sampled_y, algo_name='UMAP', target_name="DEATH_EVENT")

# %%
component_names_umap_3d_smoteenn = [f"UMAP{i+1}" for i in range(embedding_smoteenn.shape[1])]
df_umap_3d_structurized_smoteenn = pd.DataFrame(embedding_smoteenn, columns=component_names_umap_3d_smoteenn)

mi_scores_umap = make_mi_scores(df_umap_3d_structurized_smoteenn, combo_sampled_y)

plot_mi_scores(mi_scores_umap)

# %% [markdown]
# After using ***SMOTEENN*** we can see that all of the ***UMAP*** compontents provide valuable information about the target which we can clearly see on the plots above.

# %% [markdown]
# ### ***Isomap***

# %% [markdown]
# Isomap (Isometric Mapping) is a dimensionality reduction technique used to uncover the underlying structure of high-dimensional data while preserving the geometric distances between data points. It's particularly useful for visualizing complex data where nonlinear relationships exist.
# 
# Here's a breakdown of how Isomap works:
# 
# 1. Constructing a Neighborhood Graph: Isomap starts by identifying the nearest neighbors for each data point based on a distance metric (usually Euclidean distance). It creates a graph where each data point is connected to its closest neighbors.
# 2. Calculating Geodesic Distances: Instead of using straight-line distances, Isomap calculates the shortest paths (geodesic distances) between all pairs of data points in the neighborhood graph. This step helps capture the true structure of the data, accounting for any nonlinear relationships.
# 3. Multidimensional Scaling (MDS): Isomap then uses a technique called Multidimensional Scaling to embed the data into a lower-dimensional space. MDS finds a new configuration of points in a lower-dimensional space that preserves the geodesic distances as much as possible.
# 
# The final result is a low-dimensional representation of the data, where the relationships between data points are preserved based on their geodesic distances. This makes Isomap a powerful tool for visualizing and exploring the intrinsic geometry of data, especially when the data lies on a curved manifold in a higher-dimensional space.

# %%
isomap = Isomap(n_neighbors=5, n_components=3)
iso_df = isomap.fit_transform(df_preprocessed)

plot_algo3d(iso_df, df_preprocessed_with_target['DEATH_EVENT'], algo_name='Isomap', target_name="DEATH_EVENT")

# %%
component_names_isomap_3d = [f"ISOMAP{i+1}" for i in range(iso_df.shape[1])]
df_isomap_3d_structurized = pd.DataFrame(iso_df, columns=component_names_isomap_3d)

mi_scores_isomap = make_mi_scores(df_isomap_3d_structurized, df_preprocessed_with_target['DEATH_EVENT'])

plot_mi_scores(mi_scores_isomap)

# %% [markdown]
# We can see that ***ISOMAP1*** conveys the most (but still small) information, ***ISOMAP2*** conveys miniscule amount, whilst ***ISOMAP3*** conveys none.

# %%
isomap_smoteenn = Isomap(n_neighbors=5, n_components=3)
iso_smoteenn_df = isomap.fit_transform(combo_sampled_X)

plot_algo3d(iso_smoteenn_df, combo_sampled_y, algo_name='Isomap', target_name="DEATH_EVENT")

# %%
component_names_isomap_smoteenn_3d = [f"ISOMAP{i+1}" for i in range(iso_smoteenn_df.shape[1])]
df_isomap_smoteenn_3d_structurized = pd.DataFrame(iso_smoteenn_df, columns=component_names_isomap_smoteenn_3d)

mi_scores_isomap_smoteenn = make_mi_scores(df_isomap_smoteenn_3d_structurized, combo_sampled_y)

plot_mi_scores(mi_scores_isomap_smoteenn)

# %% [markdown]
# After **SMOTEENN**, ***ISOMAP1*** conveys a lot of information, ***ISOMAP2*** moderate, and ***ISOMAP3*** small amount. 
# 
# ***ISOMAP1*** retains more information from the ***SMOTEENN-sampled*** data.
# This suggests that focusing on ***ISOMAP1***, and possibly ***ISOMAP2***, would be more beneficial for understanding and interpreting the data, while ***ISOMAP3*** could potentially be discarded or given less importance.

# %% [markdown]
# ### ***t-SNE***
# 
# We'll apply t-SNE on both the SMOTEENN-sampled data and the PCA-reduced SMOTEENN-sampled data.

# %% [markdown]
# t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique primarily used for visualizing high-dimensional data in a low-dimensional space, typically 2D or 3D. It's particularly effective at revealing the structure of the data, such as clusters or complex relationships, even when these structures are non-linear.
# 
# Here's how t-SNE works:
# 
# 1. Measuring Similarities in High Dimensions: t-SNE begins by calculating the similarity between each pair of data points in the high-dimensional space. It uses a probability distribution, typically a Gaussian, to measure how similar one point is to another, focusing more on preserving the relative distances between nearby points.
# 2. Defining Similarities in Low Dimensions: In the lower-dimensional space, t-SNE again defines similarities between points, but this time using a Student's t-distribution, which has heavier tails than a Gaussian distribution. This choice helps t-SNE spread out the points more evenly and prevents them from crowding together.
# 3. Minimizing the Difference: t-SNE then adjusts the positions of the points in the low-dimensional space to minimize the difference between the pairwise similarities in the original high-dimensional space and the newly mapped space. This is done through an optimization process, ensuring that points that were close in the original space remain close in the low-dimensional representation.
# 
# The result is a map where similar data points are grouped together, and dissimilar points are far apart, making it easy to identify patterns like clusters or outliers. t-SNE is especially useful for exploratory data analysis, providing a visually intuitive way to understand the complex structure of data.

# %%
# Visualization using t-SNE
tsne = TSNE(n_components=3, random_state=42)
tsne_values = tsne.fit_transform(combo_sampled_X)

plot_algo3d(tsne_values, combo_sampled_y, algo_name='t-SNE', target_name="DEATH_EVENT")

# %% [markdown]
# Seems like the purpled data points ***(DEATH_EVENT=0)*** are more densely packed compared to the yellow data points ***(DEATH_EVENT=1)***. This is likely to be an indication of similarity between the purple data points, while the yellow ones show greater variation.
# 
# It also looks like component 3 has successfully drawn a discrimination between the two groups of data points. Although there are some overlaps, the overall separation is palpable.

# %%
component_names_tsne_3d = [f"t-SNE{i+1}" for i in range(tsne_values.shape[1])]
df_tsne_3d_structurized = pd.DataFrame(tsne_values, columns=component_names_tsne_3d)

mi_scores_tsne = make_mi_scores(df_tsne_3d_structurized, combo_sampled_y)

plot_mi_scores(mi_scores_tsne)

# %%
# Visualization using t-SNE
tsne = TSNE(n_components=3, random_state=42)
tsne_values_pca = tsne.fit_transform(df_pca_3d_smoteenn)

plot_algo3d(tsne_values_pca, combo_sampled_y, 't-SNE + PCA', target_name='DEATH_EVENT')

# %%
component_names_tsne_pca_3d = [f"t-SNE + PCA{i+1}" for i in range(tsne_values_pca.shape[1])]
df_tsne_pca_3d_structurized = pd.DataFrame(tsne_values_pca, columns=component_names_tsne_pca_3d)

mi_scores_tsne_pca = make_mi_scores(df_tsne_pca_3d_structurized, combo_sampled_y)

plot_mi_scores(mi_scores_tsne_pca)

# %% [markdown]
# ### Experiment (plotting best components of each DimRed)

# %%
df_mixed_smoteenn = pd.DataFrame(
    {'UMAP': df_umap_3d_structurized_smoteenn.iloc[:, 0],
     'Isomap': df_isomap_smoteenn_3d_structurized.iloc[:, 0],
     't-SNE + PCA': df_tsne_pca_3d_structurized.iloc[:, 0]
                                  })

plot_algo3d(df_mixed_smoteenn.values, combo_sampled_y, algo_name='Mixed', target_name="DEATH_EVENT")

# %%
mi_scores_mixed_smoteenn = make_mi_scores(df_mixed_smoteenn, combo_sampled_y)

plot_mi_scores(mi_scores_mixed_smoteenn)


