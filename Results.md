# Bias Detection Task - RESULTS

Model with better performance (evaluated using devset):
> **XGB on TF IDF Vectors, Ngram Level**

```
model = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_df=0.25)),  
        ('xgb_clf', XGBClassifier(use_label_encoder=False))])  
```

Best score (mean cross-validated score of the best_estimator) after fine-tuning: **0.502**.

**Classification report**:
  
                   precision    recall    f1-score   support

           Center       0.67      0.53        0.59        19
           Left         0.25      0.40        0.31         5
           Right        0.14      0.17        0.15         6
    
    accuracy                                  0.43        30
    macro avg           0.35      0.36        0.35        30
    weighted avg        0.49      0.43        0.45        30

<br/>
<br/>

## Experiments

All the experiments (and fine-tuning) were performed on the [bias_classifier_v3.ipynb](https://github.com/mansogf/bias_detection_task/blob/main/bias_classifier_v3.ipynb). There you can find experiments with the following models: RandomForest (RandomForestClassifier), Naive Bayes (MultinomialNB), Support Vector Machines (SVC) and XGBoost (XGBClassifier).

In terms of Feature Engineering methods, it was used 2 techniques in diferent levels. They were: TF-IDF (Term Frequency — Inverse Document Frequency) and CountVectorizer with the 'char', 'word' and 'n-grams' levels.

It's **important** to note that the datasets provided for this project contain 01 observation (id:175) that all values are 'NaN' in the test_set (*fixed in the data processing phase*), classes are not balanced ("Center" class with a proportion of approximately 2:1), and only a small number observations. This directly impacts model performance when trying to generalize the data. As an alternative to increasing performance, the 'title' and 'body' features have been concatenated ('title_body') and used for training the model.
