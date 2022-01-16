# import libraries
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

import pickle


def load_data(database_filepath):
    '''Loading the data from the provided sqlite db. Returns X, y and list of category_names '''
    
    url = 'sqlite:///' + database_filepath
    engine = create_engine(url)
    df = pd.read_sql_table('messages', engine)
    X = df[['message', 'genre']]
    y = df.iloc[:,4:]
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    '''Cleans, tokenizes and lemmatizes the provided text '''
    
    tokenised = word_tokenize(text)
    lemmatiser = WordNetLemmatizer()
    
    normalised = []
    
    for token in tokenised:
        normalised_token = lemmatiser.lemmatize(token).lower().strip()
        normalised.append(normalised_token)
        
    return normalised


def build_model():
    '''Function training the multi classifier model'''
    
    #defining pipeline for text column transformation:
    text_transformer = Pipeline(
        steps =[('vectoriser', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
           ]
        )
    
    #defining one_hot_encoder, which will be used for 'genre' column
    one_hot_transformer = OneHotEncoder()

    #defining a column transformer, to use different preprocessing estimators on 'message' and 'genre' columns
    preprocessor = ColumnTransformer(
            transformers=[
                    ("text", text_transformer, 'message'),
                    ("one_hot", one_hot_transformer, ['genre'])
                ]
            )

    #defining the classifier pipeline, using RandomForest as the model
    classifier = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ('multiclf', MultiOutputClassifier(XGBClassifier(use_label_encoder=False,
                                                                     eval_metric = 'logloss')))
                  ]
            )
    
    return classifier


def evaluate_model(model, X_test, Y_test, category_names):
    '''function printing the evaluation score of the model'''
    y_pred = model.predict(X_test)
    #turning prediction into data_frame for bette formatting
    y_pred_df = pd.DataFrame(y_pred, columns = category_names)
    
    print(y_pred.shape)
    print(y_pred_df.shape)
    for category in category_names:        
        print(category)
        print(classification_report(Y_test[category], y_pred_df[category])) 
        
    #print(classification_report(Y_test, y_pred_df, target_names = category_names))

              
              
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)



    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disasterPipeline.db classifier.pkl')


if __name__ == '__main__':
    main()