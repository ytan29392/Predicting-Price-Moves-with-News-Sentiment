import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
from nltk.tokenize import word_tokenize
from collections import Counter
import os

# Download NLTK data
nltk.download('punkt', quiet=True)


def load_data(file_path):
    return pd.read_csv(file_path)


def analyze_headline_length(df, save_path='reports/headline_length_distribution.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df['headline_length'] = df['headline'].apply(len)
    print("Headline Length Statistics:")
    print(df['headline_length'].describe())
    plt.figure(figsize=(10, 6))
    sns.histplot(df['headline_length'], bins=30)
    plt.title('Distribution of Headline Lengths')
    plt.xlabel('Headline Length (characters)')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()
    return df['headline_length'].describe()


def analyze_publishers(df, save_path='reports/publisher_counts.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    publisher_counts = df['publisher'].value_counts()
    print("Top 10 Publishers:")
    print(publisher_counts.head(10))
    plt.figure(figsize=(10, 6))
    publisher_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publishers by Article Count')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.savefig(save_path)
    plt.close()
    return publisher_counts


def analyze_publication_dates(df, save_path='reports/publication_frequency.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date_only'] = df['date'].dt.date
    date_counts = df['date_only'].value_counts().sort_index()
    print("Publication Dates Summary:")
    print(date_counts)
    plt.figure(figsize=(10, 6))
    date_counts.plot()
    plt.title('Article Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.savefig(save_path)
    plt.close()
    return date_counts


def analyze_keywords(df, save_path='reports/common_words.txt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    words = df['headline'].str.lower().apply(word_tokenize).explode()
    common_words = Counter(words).most_common(20)
    print("Top 20 Common Words:")
    print(common_words)
    with open(save_path, 'w') as f:
        f.write(str(common_words))
    return common_words


def analyze_publisher_domains(df, save_path='reports/publisher_domains.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df['publisher_domain'] = df['publisher'].str.extract(r'@([\w\.\-]+)')
    domain_counts = df['publisher_domain'].value_counts()
    print("Top 10 Publisher Domains:")
    print(domain_counts.head(10))
    plt.figure(figsize=(10, 6))
    domain_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publisher Domains')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.savefig(save_path)
    plt.close()
    return domain_counts