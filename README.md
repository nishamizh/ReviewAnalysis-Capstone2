# ReviewAnalysis-Capstone2
Review analysis - NLP

Overview:
This project is about analyzing the sentiments of Amazon Reviews and categorizing them based on their Reviews as either Postive, Neutral, Negative.

Objective: build an ML learning model that predicts sentiment type of Reviews in the dataset and suggest their price increase/ decrease/same. Whenever there is Postive Review, the product can have Price increase Flag as True. Whenever there is Negative Review, the product can have Price increase Flag as False. The consolidation of Postive and Negative review count will determine the Product Price change. 

### 1. Problem Statement: 
There are numerous products on Amazon from various sellers. The customers may or may not be satisfied with the product due to many factors any may be expressed by their reviews. This sentiment analysis may help Amazon Pricing team and also Quality improvement Team to improve their Quality based on the sentiments of the Users and helps to update their Pricing/ Quality accordingly.

Amazon has Reviews for it’s various Products. Analysis of review tells the POSITIVE / NEGATIVE / NEUTRAL rating of a Product. This highest cumulative is used by QA team to send feedback to the Sellers to either improve their Product Quality / Appropriate their Price


About the Data:
The Data is got from Kaggle, https://www.kaggle.com/datasets/yacharki/amazon-reviews-for-sa-binary-negative-positive-csv
This particular data has 3000000 rows(review details). This data(each row) has Ratings, Review Title, Reviews.

### 2. Data wrangling:
This particular data has 3000000 rows.This data has Ratings, Review Title, Reviews.
During this phase the treatment of Duplicate, missing values, data types, and other takes place specific to NLP.
Data Wrangling plays the most important role for the sentiment analysis problems. It prepares the data to the other steps susch as the EDA and Modelling. If any step is missed or done with mistake leads to wrong prediction and redoing the work again. During the course of this project I have done it several times until things were more steady in EDA.

For this Data wrangling the steps I followed:
1. I checked for missing values, except for Review_Title    0.63% which had missing values, which means it's close to zero so it can be negligible.
2. Out of 3000000, there are 188 missing values , Since the missing values are in title, it may not make difference in finding the analysis, so this can be ignored.
3. Verified if all DataTyes are in proper format and replaced them accordingly.
4. Deleted if there were duplication in Data for the same Title and Review if the contents were the same and removed them.
5. Check if there is any imbalance in the dataset for differernt target category. In this dataset there were no data imbalance.
6. Updating Rating 1 as Negative, 3 as Neutral, 5 as Postive. Scope of this project is not for RAting 2 and 4.
7. The Strings are made sure to have only Alphabets and no other special characters.
8. The Review_Title and Review are tokenized into individual words as a list in a separate column as title_word_tokenize_count and Cleaned_Review. <i> (we will utilize only Review for Sentiment analysis and use them for predicting the categories) </i>
9. Stemmer and a Word Lemmatizer are text preprocessing tools that reduce words to their base form.
10. The feature set is saved in a csv format for EDA.

### 3. EDA
The data visualization and Feature Engineering takes place and prepares the dataset for Modelling.
For this EDA the steps I followed:
1. The visualization of the Data are performed to view how the Data looks using WordCloud, where the size of each word reflects its frequency or importance, making key terms stand out at a glance.
2. The Histogram for each of the Ratings for each words count and NumberOfWords is done for each of the Rating 1, 3, 5.
3. The most frequent TOP 70 words for each of the Rating 1, 3, 5 are visualzed to know the frequency usage of various words.
3. Distribution of Review Lengths are found based on the number of words and visualized to see how it looks.
4. Lexical Diversity for each Rating and Sentiment is visualized to find their correlation.
5. PCA is used to fetch if we could obtain variance in data by few features but found to be that for achieving 90% of the variance: 93 components is needed, for achieving 95% of the variance: 251 components is needed, for achieving  ~100% of the variance: 504 components is needed. Each feature taken for consideration are the words in the Amazon review.
6. The feature set is saved in a csv format for modelling step.

| Variance Threshold | no. of Components | What It Means |
| :----------------- | :----------------: | ------------: |
| 90%                | 93                 | You can compress your 504‑dimensional space down to 93 orthogonal features while retaining most of the signal. |
| 95%                | 251                | To capture a bit more subtle variance, you need almost triple the components compared to 90%. |
| 100%               | 504                | This is the full feature set after preprocessing — no dimensionality reduction. |


### 4. Modeling
The dataset will be passed onto various ML models such as Logistic Regression, KNN, Decision tree, Random Forest, SVM.
Since it’s a Classification Problem, I would be using the below metrics to find its efficiency.

1. The Data is pre-processed first before it's used for Modelling.
2. In Data Pre-processing, the data is defined as Feature <i>(except the target and few of the redudant features all others are used as Features)</i> and Target - Rating_Sentiment`.
3. The data is then split into Training and Testing Data set where 80% of the data were used for Traiing the rest for Testing the Model performance using Supervided Learning method, to find the model Performance.
4. Traditional Machine Learning model such as Logistic Regression, KNN, Naive Bayes, Random Forest, XGBoost, Gradient Boosting were performed in Modelling.
5. Initaially all the models were performed with basic parameters and their efficieny are evaluated.
6. Then with KFold Validatation with shuffle and using hyper paramameter optimaztion techniques using GridSearchCV and HyperOpt are used to improve to model performance.

#### Logistic Regression Analysis:
Overall Accuracy: ~74.7% — decent for a 3‑class sentiment task, but not the whole story.

##### Negative class:
Precision (0.92) is excellent — when the model predicts “Negative,” it’s almost always correct.

Recall (0.83) is strong but slightly lower, meaning some negatives are missed.

##### Neutral class:
Precision (0.64) and Recall (0.56) are the weakest — the model struggles to correctly identify and capture neutral cases, likely confusing them with Positive or Negative.

##### Positive class:
Precision (0.69) is moderate, but Recall (0.85) is high — it catches most positives, but with more false positives than the Negative class.

#### Naive Bayes
Overall Accuracy: ~64.3% — notably lower than the ~74.7% you saw with Logistic Regression and KNN.

##### Negative class:
Precision (0.90) is still high — when it predicts “Negative,” it’s usually right.
Recall (0.73) dropped compared to LR/KNN, meaning more negatives are being missed.

##### Neutral class:
Precision (0.57) and recall (0.50) are both weak — the model struggles to both identify and correctly classify neutral cases.

##### Positive class:
Precision (0.53) is low, but recall (0.70) is higher — it’s catching many positives but with a lot of false positives.

🔍 Inference
1. Performance drop is expected — Multinomial NB assumes feature independence and works best when features are raw counts or TF‑IDF with strong class‑conditional word distributions.
2. The gap between Negative and Positive precision suggests the model is more confident in detecting strong negative cues than positive ones.
3. Neutral remains the hardest class — consistent with your other models — but here the gap is even wider.

💡 Why Naive Bayes is underperforming here
* Linear separability: Your earlier models (LR/KNN) likely benefited from richer decision boundaries, while NB’s independence assumption limits flexibility.
* Feature scaling: NB doesn’t use distance metrics, but if your TF‑IDF features are sparse and high‑dimensional, LR/KNN can adapt better.
* Class overlap: NB struggles when word usage overlaps heavily between classes — which is common for Neutral vs. Positive/Negative.


### Success Metrics:  
Accuracy, Precision, Recall, F1 score, ROC-AUC 

### Possible Insights: 
Analyze the product reviews and predict the sentiments of the Reviewers.
Develop a Score prediction based on their sentiments and recommend either Product Price to increase/decrease/same.

### Criteria for success
1. Improve the Quality of the Product
2. Decrease/ Increase/ No change in the Price as per the Review Analysis.
3. Customer Satisfaction is the highest for the Product.


