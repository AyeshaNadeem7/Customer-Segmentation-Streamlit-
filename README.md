## Customer Segmentation Project:

**1. Introduction**

Customer segmentation is a key business strategy that helps companies divide their customer base into distinct groups based on demographic and behavioral data. By identifying unique customer groups, businesses can design targeted marketing campaigns, improve customer experiences, and increase profitability.

In this project, we used the Mall Customers dataset to segment customers based on their demographic attributes (Age, Gender) and financial/behavioral features (Annual Income and Spending Score). Two clustering methods were applied: K-Means Clustering and Hierarchical Clustering.

**2. Approach**

**Data Loading and Cleaning:**

The dataset was loaded from a CSV file containing 200 customer records.

**Features:**

CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).

Non-numerical features (e.g., Gender) were not used in clustering to avoid bias.

**Exploratory Data Analysis (EDA):**

Distribution of gender, age, annual income, and spending score was visualized.

Found a balanced gender split, diverse age groups, and wide variation in income/spending patterns.

**Feature Scaling:**

StandardScaler was applied to normalize numerical features (Age, Annual Income, and Spending Score) for better clustering performance.

**Clustering Algorithms:**

K-Means Clustering: Tested for different cluster values (k=2 to 10). Silhouette score was used to determine the best fit.

Hierarchical Clustering: Applied with different linkage methods (ward, complete, average, single). A dendrogram was generated to visualize natural divisions in the data.

**Cluster Visualization:**

2D scatter plots (Income vs. Spending Score) were created for both clustering methods.

Customers were segmented into four clusters, with meaningful business interpretations.

**3. Challenges**

Parameter Tuning: Choosing the correct number of clusters required careful analysis using the Elbow Method and Silhouette Score.

Interpretability: Translating numerical clusters into meaningful business insights required domain understanding.

Model Updates: In the latest versions of scikit-learn, the affinity parameter in Hierarchical Clustering was replaced with metric, requiring adjustments to the implementation.

**4. Outcomes**

The clustering identified four distinct customer segments:

Cluster 0 – Low income, low spenders → Less profitable segment.

Cluster 1 – High income, high spenders → Best customers, strong target for premium offers.

Cluster 2 – Low income, high spenders → Price-sensitive but loyal, respond well to discounts.

Cluster 3 – High income, low spenders → Upselling opportunity (encourage higher engagement).

Business Value:

Enables targeted marketing campaigns (e.g., discounts for Cluster 2, premium offers for Cluster 1).

Helps in customer retention by identifying loyalty segments.

Guides strategic decisions for product placement, promotions, and personalized offers.

**5. Conclusion**

This project demonstrated the effectiveness of unsupervised learning techniques (K-Means and Hierarchical Clustering) in segmenting mall customers into meaningful groups. Despite challenges in parameter tuning and algorithm compatibility, the final models provided clear and actionable insights that businesses can leverage to optimize customer engagement and profitability.

