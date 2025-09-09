# ReviewAnalysis-Capstone2
Review analysis - NLP

Overview:
This project is about analyzing the sentiments of Amazon Reviews and categorizing them based on their Reviews as either Postive, Neutral, Negative.

Objective: build an ML learning model that predicts sentiment type of Reviews in the dataset and suggest their price increase/ decrease/same. Whenever there is Postive Review, the product can have Price increase Flag as True. Whenever there is Negative Review, the product can have Price increase Flag as False. The consolidation of Postive and Negative review count will determine the Product Price change. 

### 1. Problem Statement: There are numerous products on Amazon from various sellers. The customers may or may not be satisfied with the product due to many factors any may be expressed by their reviews. This sentiment analysis may help Amazon Pricing team and also Quality improvement Team to improve their Quality based on the sentiments of the Users and helps to update their Pricing/ Quality accordingly.

Amazon has Reviews for it’s various Products. Analysis of review tells the POSITIVE / NEGATIVE / NEUTRAL rating of a Product. This highest cumulative is used by QA team to send feedback to the Sellers to either improve their Product Quality / Appropriate their Price


About the Data:
The Data is got from Kaggle, https://www.kaggle.com/datasets/yacharki/amazon-reviews-for-sa-binary-negative-positive-csv
This particular data has 3000000 rows(review details). This data(each row) has Ratings, Review Title, Reviews.

### 2. Data wrangling:
This particular data has 3000000 rows.This data has Ratings, Review Title, Reviews.
During this phase the treatment of Duplicate, missing values, data types, and other takes place specific to NLP.
Data Wrangling plays the most important role for the sentiment analysis problems. It prepares the data to the other steps susch as the EDA and Modelling. If any step is missed or done with mistake leads to wrong prediction and redoing the work again. During the course of this project I have done it several times until things were more steady in EDA.

For this particular project the steps I followed was
1. I checked for missing values, except for Review_Title    0.63% which had missing values, which means it's close to zero so it can be negligible.
2. Out of 3000000, there are 188 missing values , Since the missing values are in title, it may not make difference in finding the analysis, so this can be ignored.
3. Verified if all DataTyes are in proper format and replaced them accordingly.
4. Deleted if there were duplication in Data for the same Title and Review if the contents were the same and removed them.
5. Check if there is any imbalance in the dataset for differernt target category. In this dataset there were no data imbalance.
6. Updating Rating 1 as Negative, 3 as Neutral, 5 as Postive. Scope of this project is not for RAting 2 and 4.
7. The Strings are made sure to have only Alphabets and no other special characters.
8. The Review_Title and Review are tokenized into individual words as a list in a separate column as title_word_tokenize_count and Cleaned_Review. <i> (we will utilize only Review for Sentiment analysis and use them for predicting the categories) </i>
9. Stemmer and a Word Lemmatizer are text preprocessing tools that reduce words to their base form.
10. TF‑IDF vectorizer is used to convert text into numerical feature vectors for Model Building.
11. At the end of the Data Wrangling the number features in the dataset is 116, where we initially started with only 3 features. Each Word text is tokenized and TF‑IDF vectorized to get 116 features.

### 3. EDA
The data visualization and Feature Engineering takes place and prepares the dataset for Modelling.

### 4. Modeling
The dataset will be passed onto various ML models such as Logistic Regression, KNN, Decision tree, Random Forest, SVM.
Since it’s a Classification Problem, I would be using the below metrics to find its efficiency.


### Success Metrics:  Accuracy, Precision, Recall, F1 score, ROC-AUC 

### Possible Insights: 
Analyze the product reviews and predict the sentiments of the Reviewers.
Develop a Score prediction based on their sentiments and recommend either Product Price to increase/decrease/same.

### Criteria for success
Improve the Quality of the Product
Decrease/ Increase/ No change in the Price as per the Review Analysis.
Customer Satisfaction is the highest for the Product.


