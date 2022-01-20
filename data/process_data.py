import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Function loading the .csv datasets from specified paths and comning it into singl data frame'''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = categories.merge(messages, on = 'id', how = 'inner')

    return df

def clean_data(df):
    '''
    Function cleaning the combined datasets.
    It creates and cleans new columns based on the categories file and removes all duplicates
    '''
    
    # categories splitting into separate columns:
    categories = df['categories'].str.split(';', expand = True)
    
    #creating column names based on the first part of the string 'XXX_Y'
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    #saving the column values as the numeric, which is the last character in the orignial string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # in some cases the columns had value '2' instead of just 0/1. Replacing those occurances with 1 
        categories[column] = categories[column].apply(lambda x: 1 if x == 2 else x)
    
    #dropping the original categories column and joining the new wide version
    df = df.drop('categories', axis = 1)
    df = df.join(categories)
    
    #dropping duplciates based on the id column    
    df = df.drop_duplicates(subset = 'id')
    
    return df
    
def save_data(df, database_filename):
    ''' Function saving the clean dataset to a specified sqlite database'''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    print(df.shape)
    df.to_sql('messages', engine, index=False, if_exists= 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disasterPipeline.db')


if __name__ == '__main__':
    main()