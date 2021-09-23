# Bias detection task

The goal of this task is to train a classifier to detect political bias in news articles.

# Data

The data folder contains three files

- bias_articles_train.json: Json containing array of dicts with 
  - `id`: unique integer id for each article
  - `title`: Title of the article
  - `body`: Body of the article
  - `bias`: Class label it can be "Left", "Center" and "Right"
- bias_articles_dev.json: Same as bias_articles_train, but used for validation of the models.
- bias_articles_test.json: Same as bias_articles_train, but does not contain bias labels (to be predicted.)

# Task
- In bias_classifier.py Implement fit, test and predict functions.
- You may use any classifier that you like, any features extracted from the title and the body.
- Use bias_articles_train.json to train the model, bias_articles_dev.json to fine-tune the parameters.
- Predict the labels for articles in bias_articles_test.json using the model trained above. 
- Document the code and follow [Google python coding conventions](https://google.github.io/styleguide/pyguide.html)
- Document the results of your model (train performance vs test performance) in Results.md markdown file.
- Fork this repository and implement the solution and make a pull request.
- If you have any question keep in touch with Vinay Setty (first name at factiverse.no).
