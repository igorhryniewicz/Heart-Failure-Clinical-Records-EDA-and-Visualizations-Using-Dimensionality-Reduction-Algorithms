# *Heart Failure Clinical Records* - Comprehensive ***EDA*** and Visualisations With Usage of Dimensionality Reduction Algorithms

This project is an elective part of university's course about Multivariate Data Analysis, in order to enhance ***EDA*** skills and its optimal usage to understand and consequently visualise the data. [More information here.](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

## Part I - ***EDA*** Before Cleaning

Firstly we noticed there are no null values. While checking the data types, we have found that $2$ features - *age*, *platelets* are **float**, when they should logically be of **int** type. Moreover, there are no categorical features.

The dataset shows a correlation between indices and *time*, likely because the data is sorted by the *time* feature. 

Several features, such as *creatinine_phosphokinase*, *platelets*, and *serum_creatinine*, exhibit right-skewed distributions and contain outliers.

Particularly, *creatinine_phosphokinase* has a noticeable difference between its **mean** and **median**, suggesting skewness, while *platelets* and *serum_creatinine* have extremely high **maximum** values indicating outliers.

The target feature, *DEATH_EVENT*, is imbalanced, with $32\%$ of cases indicating death from heart failure. Violin plots have confirmed the presence of outliers across multiple features and showed significant differences in *time* and *ejection_fraction* distributions with respect to *DEATH_EVENT*.

We explored transformations, such as $\log(1+x)$, on skewed features like *creatinine_phosphokinase* and *serum_creatinine*, concluding that *creatinine_phosphokinase* is not normally distributed. 

There are no significant correlations between features of interest. Additionally, *platelets* and *age* can be converted from **float** to **int** after rounding. The data is notably imbalanced, with nearly twice as many samples labeled as $0$ compared to $1$, reinforcing the imbalance in the target variable.

## Part II - Cleaning

We got rid of outliers using ***IQR*** method, and duplicates. Number of rows dropped from $299\to 251$.

## Part III - ***EDA*** After Cleaning

After cleaning, the dataset no longer exhibits the prominent outliers that were previously observed. The distributions have smaller tails, indicating a reduction in extreme values while still retaining significant high and low values.

The features *time*, *serum_creatinine*, and *age* show significant differences in mean with respect to *DEATH_EVENT*, suggesting a strong relationship with the target variable. 

Our earlier hypotheses about the normality of the data are likely incorrect, as normal distributions are not evident. Applying the $\log(1+x)$ transformation again did not yield significant changes in the data distribution. 

Only features ranked above *sex* show meaningful significance in ***MI*** score in relation to the target variable.

## Part IVa - Statistical Tests

Every test was conducted with significance level $\alpha = 0.05$.

After conducting the $\chi^2$ independence test on the binary data, we found that *diabetes* and *sex*, as well as *smoking* and *sex*, are dependent on each other, with a moderate strength of association observed in the relationship between *smoking* and *sex*.

***The Shapiro-Wilk*** test indicated that none of the features follow a normal distribution, a finding further supported by the ***Q-Q*** plots. 

Despite *creatinine_phosphokinase* having a non-zero ***Mutual Information (MI)*** score, its mean does not significantly differ when conditioned on *DEATH_EVENT*, suggesting limited predictive value. The results from the ***ANOVA*** tests on each non-binary feature with respect to a binary one align with the outcomes from the mutual information analysis, confirming the significance of certain features.

## Part IVb - Experimental Statistical Tests

In this part of the analysis, we converted non-binary features into categorical data based on our prior conclusions and relevant medical information. 

Acknowledging that human physiology doesn't always align perfectly with medical standards, these categorizations should be interpreted cautiously.

Considering insights from ***Violin Plots***, ***Mutual Information***, and ***ANOVA*** tests, we identified *time*, *serum_creatinine*, *age*, *serum_sodium*, and *ejection_fraction* as particularly significant in relation to *DEATH_EVENT*. These features exhibited clear prominence and importance in our analysis, reinforcing their relevance to the target variable.

## Part V - Preprocessing

Considering the significance of *age*, *ejection_fraction*, *serum_creatinine*, *serum_sodium* and *time* we decided to use only these features for our dimensionality reduction models. Moreover, we scaled the data in order for features with high values not to overtake.

As noted in the ***EDA*** section, the dataset initially had a class imbalance, with approximately $68\%$ of instances belonging to one class. This imbalance has been addressed through various techniques:

- Over-sampling: Synthetic examples were created in the minority class using ***SMOTE***.
- Under-sampling: Instances from the majority class were deleted or merged using the ***Random Sampler***.
- Combination of both: ***SMOTEENN*** was employed, combining over-sampling with ***SMOTE*** and under-sampling with ***Edited Nearest Neighbors (ENN)***.

Based on our analysis using ***LDA***, ***SMOTEENN*** proved to be the most effective method for addressing class imbalance. This approach resulted in the clearest distinction between classes, making it the preferred choice for improving dataset balance and model training suitability.

## Part VI - Dimensionality Reduction Models

We used $4$ dimensionality reduction methods - ***PCA***, ***UMAP***, ***Isomap*** and ***t-SNE***. We used $3$ components in order to visualise data more clearly. 

Then, we calculated ***MI*** scores to check the viability of the components for the target. We did it $2$ times for each model - **with** and **without** ***SMOTEENN***.

The results were unambiguous. Thanks to ***SMOTEENN*** we achived significantly better ***MI*** scores, and separability on the graphs.

Additionaly we can see consistency in visualisation for each model - people who live are scattered more densly and close to each other suggesting existence of a 'safe zone', whereas people who died of heart failure divided into $2$ smaller clusters, which could mean $2$ possible extremums in parameters that contribute to their death.

Moreover, when the patient is still being treated, *time* feature should not be considered as there is a tendency that the when its value is low - person dies, and when high - person lives.

