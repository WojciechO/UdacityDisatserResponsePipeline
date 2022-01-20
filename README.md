# Disaster Response Pipeline Project

### Overview
This project is part of Udacity Data Science Nanodegree program.
It includes following steps:
* a data pipeline, reading in data, cleaning it and saving to the SQLite database
* ML pipeline: running NLP transformations, training  XGBoost multi-output classifier and saving it as pickle
    * GridSearch cross validation was performed off line to tune the hyperparamters
* A flask app:
    * displaying two visualisations made with plotly
    * allowing for user message to be classified

### Prerequisites
Following libraries are not part of standard anaconda installation and need to be installed:

* scikit-learn with version > 0.20.0 (it is required for ColumnTransaformer)
* xgboost
* nltk
* sqlalchemy
* plotly
* flask


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disasterPipeline.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disasterPipeline.db models/classifier.pkl`
	- Training the XGBoost model may take a while. The already trained model is present in the model/classifier.pkl file

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Notes on training and performance

After training and comparing performance of Random Forest classifier and XGBoost classifier, the latter was selected as it improved the recall. XGBoost model was also able to predict occurances of some of the classes with the small support. Random Forest model had 6 classes with low support, that were never predicted. XGB did not make any predicton for only 2 classes

Running grid search to tune hyperparameters found the following values to perform the best.
XGBoost:
* default parameters
* learning rate lowered from default 0.3 to 0.2 (learning_rate= 0.2)

TFIDF:
* including bigrams, apart from unigrams also improved the model (ngram_range= (1, 2))


'Genre' column was used as an additional feature. It is one-hot encoded into 3 columns. To enable one-hot encoding on one column and text processing with TFiDF transformation on another, the sklearn ColumnTransformer was used.

For the user input predictions, a default genre 'Direct' is provided to the model.

