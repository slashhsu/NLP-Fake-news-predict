# NLP Fake News Prediction

## Overview
This project focuses on detecting **fake news** using **Natural Language Processing (NLP)** techniques. The dataset was sourced from **Kaggle**, and various machine learning models were implemented to classify news as real or fake.

---

## 1. Dataset
The dataset used for this project was obtained from **Kaggle** and contains labeled news articles categorized as either **fake** or **real**. The goal is to build a model that can accurately classify news articles based on their textual content.

![Dataset Overview](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/967edcc3-01d4-4173-9fab-8b79903d5484)

---

## 2. Data Preprocessing
To prepare the dataset for machine learning, the following preprocessing steps were performed:
- **Tokenization** – Splitting text into individual words
- **Stopword Removal** – Removing common words that do not add much meaning
- **Stemming/Lemmatization** – Converting words to their root form
- **TF-IDF Vectorization** – Converting text into numerical features

![Data Preprocessing Steps](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/42c12f25-1009-4688-a5eb-ab2f82e8ac2c)

---

## 3. Model Implementation
Several machine learning models were trained and evaluated for fake news detection:

- **Logistic Regression**
- **Naïve Bayes Classifier**
- **Support Vector Machines (SVM)**
- **Random Forest Classifier**
- **Deep Learning (LSTM-based model)**

![Model Overview](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/308dcdc8-123a-4800-9607-c27adec696d2)

---

## 4. Feature Selection & Model Performance
Each model was evaluated based on accuracy, precision, recall, and F1-score. The **confusion matrix** and **ROC curve** were used to assess classification performance.

![Model Performance](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/f636d822-787f-4eff-9b0d-10c36250e37f)

### **Accuracy Comparison**
Comparison of model accuracy scores:

![Accuracy Comparison](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/73a4ba69-f8c9-457e-8eab-090868548b63)

### **Confusion Matrix**
A confusion matrix showing correct and incorrect predictions:

![Confusion Matrix](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/ed777409-bbc6-4e04-a373-5d3a92a0efe2)

### **ROC Curve**
The ROC curve visualising model performance:

![ROC Curve](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/e346d618-948e-4f6f-9bc4-e37142704f04)

---

## 5. Key Insights
📌 **Preprocessing significantly improves model performance**
📌 **TF-IDF vectorization works well for text-based classification**
📌 **Deep learning models (LSTM) outperform traditional models in text classification**
📌 **Ensemble methods such as Random Forest provide robust classification**

![Key Insights](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/332a6b8e-aed1-4025-88ff-0e96bdf2c85f)

---

## 6. Future Work
🔹 **Try BERT-based models for improved accuracy**
🔹 **Explore additional NLP feature engineering techniques**
🔹 **Experiment with more sophisticated deep learning architectures**

![Future Work](https://github.com/slashhsu/NLP-Fake-news-predict/assets/137000188/a755ec1f-797f-4f17-a9dc-58a81d91e789)

---

📧 For any questions, feel free to contact me!

---

## 8. References
- Kaggle Fake News Dataset
- NLP and Machine Learning Techniques for Text Classification
- Research papers on Fake News Detection using NLP









