# Restaurant Review Sentiment Predictor

## Overview
This project aims to predict the sentiment of a restaurant review, classifying it as either **positive** or **negative**. The model processes text data using Natural Language Processing (NLP) techniques like stopword removal and stemming. The **Bag of Words** technique is applied for vectorization, converting the text data into a form that can be understood by machine learning algorithms. A **Naive Bayes** classifier is trained on the processed data to predict sentiment.

The application allows users to input their own restaurant reviews, and the model predicts whether the review is positive or negative in real-time.

## Project Structure

### 1. Data Preprocessing and Feature Engineering
- **Text Cleaning**: The reviews are cleaned by removing non-alphabetical characters, converting the text to lowercase, and splitting it into individual words.
- **Stopword Removal**: Common stopwords (e.g., "the", "is", "in") are removed from the reviews.
- **Stemming**: Words are reduced to their root form using the Porter Stemmer.
- **Vectorization**: The processed reviews are converted into a matrix of token counts using the **Bag of Words** model, with the top 1500 words extracted as features.

### 2. Model Building
- The **Naive Bayes** classifier is used to train the model. This probabilistic classifier is well-suited for text classification tasks.
- **Training**: The classifier is trained on a labeled dataset of restaurant reviews to predict whether the sentiment is positive or negative.

### 3. Model Evaluation
- **Confusion Matrix**: The performance of the model is evaluated using a confusion matrix, which shows the number of true positives, true negatives, false positives, and false negatives.
- **Accuracy**: The model's classification accuracy is assessed using various metrics, including accuracy score.

### 4. Real-Time Sentiment Prediction
- The application accepts user input, processes the review text in real-time, and predicts whether the sentiment is positive or negative.

## Dependencies Installation

To run the project locally, install the necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Required Libraries:
- numpy
- pandas
- matplotlib
- nltk
- sklearn

## Running Project Locally

### Steps to Execute

#### 1. Text Preprocessing
- Run the code to preprocess the dataset (`Restaurant_Reviews.tsv`).
- This includes cleaning, stopword removal, stemming, and vectorization.

#### 2. Model Training
- Train the **Naive Bayes** classifier using the processed dataset.

#### 3. Real-Time Sentiment Prediction
- Allow users to input their own review text.
- The system will preprocess the text, vectorize it, and predict whether the review is positive or negative.

### Example Usage:

To predict the sentiment of a new review, enter the following code:

```python
text = input('Enter the comment: ')
```
The model will then predict the sentiment and output one of the following messages:

- **Positive review given by the person**
- **Negative review given by the person**

### Example:

**Input**: "The food was amazing, and the service was great!"  
**Output**: "Positive review given by the person"

**Input**: "The food was awful, and the staff was rude."  
**Output**: "Negative review given by the person"

## Data Source
The dataset used in this project is a collection of restaurant reviews (`Restaurant_Reviews.tsv`). The dataset includes reviews and their corresponding sentiment labels (positive or negative).

## Contact
For any questions or feedback, please contact **Ritik Suri** at [Ritik1704@gmail.com](mailto:Ritik1704@gmail.com).

