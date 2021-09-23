from typing import Tuple, List, Dict
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from data_prep import data_cleaning

class BiasClassifier:
    def __init__(self):
        self._model = None

    def fit(self, train_file_path: str):
        """Train a classifier model after reading and extracting features from
        train_file_path.

        Args:
            train_file_path: String path to the training data as a json, you
            may assume instances have labels.

        Returns:
            A tuple of list of document id and prediction label and
        """

        train_df = pd.read_json(train_file_path)

        # drop id=175 (all values are NaN)
        train_df = train_df.dropna().reset_index()

        train_title_body = train_df['title'] + ' ' + train_df['body']
        train_df['title_body'] = [data_cleaning(i) for i in train_title_body]

        train_x = train_df['title_body']
        train_y = LabelEncoder().fit_transform(train_df['bias'])

        self._model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_df=0.25)),
            ('xgb_clf', XGBClassifier(use_label_encoder=False))])

        self._model.fit(train_x, train_y)

        return self._model

    def eval(self, dev_file_path: str) -> Tuple[List[Dict[str, float]], classification_report]:
        """Evaluates the test data given in test_file_path after reading and
         extracting features.

        Args:
            test_file_path: String path to the test data, you may assume
            instances have labels.

        Returns:
            A tuple of list of document id and prediction label and
            evaluation summary in the form of sklearn classification_report.
        """

        dev_df = pd.read_json(dev_file_path)

        dev_title_body = dev_df['title'] + ' ' + dev_df['body']
        dev_df['title_body'] = [data_cleaning(i) for i in dev_title_body]

        dev_x = dev_df['title_body']
        dev_y = LabelEncoder().fit_transform(dev_df['bias'])

        dev_pred = self._model.predict(dev_x)

        dev_df['bias_predicted'] = dev_pred
        dev_df["bias_predicted"].replace({0: "Center", 2: "Right", 1: "Left"}, inplace=True)

        dev_df_pred = dev_df[['id', 'bias_predicted']].to_dict('records')

        return (dev_df_pred, classification_report(dev_pred, dev_y))


    def predict(self, test_file_path: str) -> List[Dict[str, float]]:
        """Evaluates the test data given in test_file_path after reading and
         extracting features.

        Args:
            test_file_path: String path to the test data, the instances do not
            have labels.

        Returns:
            A list of document id and prediction label.
        """

        test_df = pd.read_json(test_file_path)

        test_title_body = test_df['title'] + ' ' + test_df['body']
        test_df['title_body'] = [data_cleaning(i) for i in test_title_body]

        test_x = test_df['title_body']

        test_pred = self._model.predict(test_x)


        test_df['bias_predicted'] = test_pred
        test_df["bias_predicted"].replace({0: "Center", 2: "Right", 1: "Left"}, inplace=True)

        test_df_pred = test_df[['id', 'bias_predicted']].to_dict('records')

        return test_df_pred

if __name__ == "__main__":

    bias_cls = BiasClassifier()

    bias_fit = bias_cls.fit("./data/bias_articles_train.json")
    print()
    print()
    print("==============================Training...==============================")
    print()
    print()
    print(bias_fit)
    print()
    print()
    print("==============================Evaluating...==============================")
    print()
    print()
    bias_eval = bias_cls.eval("./data/bias_articles_dev.json")
    print(bias_eval)
    print()
    print()
    print("==============================Predicting...==============================")
    print()
    print()
    bias_pred = bias_cls.predict("./data/bias_articles_test.json")
    print(bias_pred)
    print()
    print()
