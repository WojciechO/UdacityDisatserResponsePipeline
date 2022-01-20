import pandas as pd
import numpy as np

from plotly.graph_objs import Box
import plotly.graph_objects as go
from plotly.colors import n_colors

from sklearn.preprocessing import MinMaxScaler

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def remove_percentile(column, percentile = .95):
    '''Takes pandas series as an input and returns that series below specified quantile threshold '''
    
    percentile_cutoff = column.quantile(percentile)
    return column[(column < percentile_cutoff)]



def prepare_boxplot(df, column_names, target):
    '''Function transforming df columns and returning them as list of plotly Boxplot objects '''
    boxplots = []

    for column in column_names:
        target_column = df[df['genre'] == column][target]
        target_column = remove_percentile(target_column)
        new_box = Box(y = target_column,
                      name = column,
                      boxpoints = False)
        boxplots.append(new_box)

    return boxplots

def tokenize(text):	
    '''Function splitting input text into tokens and lemmatizing it '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def prepare_table(df, column_names):
    '''Function preapring and returning a plotly Table object'''	

    #creating a gropued and transposed data frame with counts of category occurances per genre
    grouped_df = df.groupby('genre').agg('sum').drop('id', axis = 1).transpose().reset_index()
    grouped_df.rename(columns = {'index':'Category'}, inplace = True)


    #preparing a colour map for the table
    colors = n_colors('rgb(245,255,250)', 'rgb(50,205,50)', 5, colortype='rgb')

    #creating a standardised values of the main df, to map colours
    grouped_df_standardised = pd.DataFrame( MinMaxScaler().fit_transform(grouped_df[column_names]),
	                                        columns = column_names)

    #mapping the colours to the standardised values
    bins = [-0.1, 0.1, .25, .5, .75, 1]
    category = [0,1,2,3,4]

    for column in grouped_df_standardised.columns:
        grouped_df_standardised[column] = pd.cut(grouped_df_standardised[column], bins, labels=category)

    #creating the table object    
    table = go.Table(
        header=dict(values=list(grouped_df.columns),
                        line_color='white', fill_color='white',
                        align='center',
                    font=dict(color='black', size=12)),
        
        cells=dict(values=[grouped_df['Category'], grouped_df.direct, grouped_df.news, grouped_df.social],
                   fill_color=['lavender',
                               np.array(colors)[grouped_df_standardised['direct']],
                              np.array(colors)[grouped_df_standardised['news']],
                              np.array(colors)[grouped_df_standardised['social']]
                              ],
                   align='left'))
    
    return table
