import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

path = '/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignments 3/assignment3_repo/ds4se3-group6/'


def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_source_code(text):
    return re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)

def preprocess_text(text):
    text = clean_html(text)
    text = remove_source_code(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def load_data_from_excel(filepath, usecols=None):
    df = pd.read_excel(filepath, usecols=usecols)
    df.columns = df.columns.str.strip()  
    return df

def load_rtf_data(filepath):
    with open(filepath, 'r') as file:
        rtf_content = file.read()
    return re.findall(r'\b\d+\b', rtf_content)

def custom_fillna(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)  
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].fillna('null')
    return df


df_posts = load_data_from_excel('  /datasets/stack_posts_data.xlsx',usecols=['QuestionId', 'QuestionTitle', 'QuestionBody', 'AnswerId', 'AnswerBody', 'AnswerScore'])
df_urls = load_data_from_excel(' /assets/Architectural_Posts.xlsx')

df_urls['Post_ID'] = df_urls['URL'].apply(lambda x: x.split('/')[-1])

df_posts['preprocessed_Text'] = df_posts.groupby('QuestionId').apply(
    lambda x: preprocess_text(f"{x['QuestionTitle'].values[0]} {x['QuestionBody'].values[0]} {x.loc[x['AnswerScore'].idxmax(), 'AnswerBody']}")
).reset_index(level=0, drop=True)

df_posts['Post_ID'] = df_posts['QuestionId'].apply(str)
merged_df = df_posts.merge(df_urls, on='Post_ID', how='outer')


rtf_ids = load_rtf_data(' /assets/Programming_posts.rtf')
merged_df['Programming_post'] = merged_df['Post_ID'].isin(rtf_ids)
merged_df['preprocessed_Text'] = merged_df['preprocessed_Text'].astype(str)
merged_df['Term_Count'] = merged_df['preprocessed_Text'].apply(lambda x: len(word_tokenize(x)))
merged_df['Post_Type'] = merged_df['Programming_post'].apply(lambda x: 'Programming' if x else 'Architectural')
merged_df['Purpose'] = merged_df['Purpose'].fillna('null')
merged_df['Solution'] = merged_df['Solution'].fillna('null')
merged_df = custom_fillna(merged_df)
merged_df.to_excel(' /datasets/preprocessed_data_new.xlsx', index=False)


print("Data preprocessing completed and saved to preprocessed_data.xlsx")
