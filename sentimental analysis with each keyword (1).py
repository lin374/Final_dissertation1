#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk


nltk.download('punkt')


pdf_path = r"C:\Users\LINCY\Downloads\Sustainabilityreport2017.pdf"  # Update with your file path
doc = fitz.open(pdf_path)


text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    text += page.get_text()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text


cleaned_text = clean_text(text)


sentences = sent_tokenize(cleaned_text)


recovery_keywords = [
    "group", "volkswagen", "sustainability", "report", "management", "production",
    "brands", "environmental", "new", "strategy", "human", "employees",
    "development", "information", "corporate", "customer", "vehicles", 
    "customers", "reporting", "business", "materials", "year", "emissions", 
    "rights", "company", "responsibility", "sustainable", "strategic"
]


analyzer = SentimentIntensityAnalyzer()


keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in recovery_keywords}

for sentence in sentences:
    sentiment = analyzer.polarity_scores(sentence)
    for keyword in recovery_keywords:
        if keyword in sentence:
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1


for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']


print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")
    else:
        print(f"{keyword.capitalize()} - No relevant sentences found.")


# In[3]:


import fitz  # PyMuPDF
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('punkt')
nltk.download('stopwords')


pdf_path = r"C:\Users\LINCY\Downloads\Nonfinancial_Report_2018_e.pdf"  # Use the correct path to your file
doc = fitz.open(pdf_path)


relevant_sections = ["Corporate Governance", "Sustainability Management", "ESG Performance Management", "Decarbonization", "Integrity", "Recovery Strategies"]


text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    page_text = page.get_text()
    if any(section in page_text for section in relevant_sections):
        text += page_text


def clean_text(text):
    
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove standalone single letters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation but keep words
    
    # Lowercase the text
    text = text.lower()

    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize stopwords
    stop_words = set(stopwords.words('english'))
    
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha() and word not in stop_words]
        cleaned_sentence = ' '.join(words)
        cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences


cleaned_sentences = clean_text(text)


keywords = ["group", "volkswagen", "sustainability", "report", "management", "production", "brands", "environmental", "new", "strategy", "human", "employees", "development", "information", "corporate", "customer", "vehicles", "customers", "reporting", "well", "business", "materials", "year", "emissions", "rights", "company", "responsibility", "sustainable", "strategic"]


analyzer = SentimentIntensityAnalyzer()


keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

for sentence in cleaned_sentences:
    for keyword in keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1


for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']


print("\nSentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[4]:


import fitz  # PyMuPDF
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('punkt')
nltk.download('stopwords')


pdf_path = r"C:\Users\LINCY\Downloads\2022_Sustainability_Report.pdf"
doc = fitz.open(pdf_path)


text = ""
relevant_sections = ["Corporate Governance", "Sustainability in the Group's DNA", "NEW AUTO Group Strategy", "Environmental Compliance Management", "ESG Performance Management", "Integrity and Decarbonization"]

for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    page_text = page.get_text()
    if any(section in page_text for section in relevant_sections):
        text += page_text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text


cleaned_text = clean_text(text)


sentences = sent_tokenize(cleaned_text)


keywords = [
    "group", "volkswagen", "report", "management", "sustainability", "compliance", 
    "risk", "business", "company", "rights", "corporate", "mobility", "reporting", 
    "environmental", "disclosures", "board", "assurance", "combined", "chain", 
    "risks", "new", "human", "information", "integrity", "supply", "companies"
]

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

for sentence in sentences:
    for keyword in keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculate average sentiment scores for each keyword
for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

# Print the sentiment analysis results for each keyword
print("\nSentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[5]:


import fitz  # PyMuPDF
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('punkt')
nltk.download('stopwords')


pdf_path = r"C:\Users\LINCY\Downloads\Sustainability_Report.pdf"
doc = fitz.open(pdf_path)


relevant_sections = ["Corporate Governance", "Sustainability Management", "ESG Performance Management", "Decarbonization", "Integrity", "Recovery Strategies"]


text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    page_text = page.get_text()
    if any(section in page_text for section in relevant_sections):
        text += page_text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text


cleaned_text = clean_text(text)


sentences = sent_tokenize(cleaned_text)


keywords = [
    "group", "volkswagen", "management", "report", "compliance", "integrity", 
    "sustainability", "emissions", "vehicles", "reporting", "business", "measures", 
    "decarbonization", "risks", "risk", "brands", "environmental", "nonfinancial", 
    "company", "target", "production", "strategy", "key"
]


analyzer = SentimentIntensityAnalyzer()


keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

for sentence in sentences:
    for keyword in keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculate average sentiment scores for each keyword
for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

# Print the sentiment analysis results for each keyword
print("\nSentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[6]:


import fitz  # PyMuPDF
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('punkt')
nltk.download('stopwords')


pdf_path = r"C:\Users\LINCY\Downloads\Nonfinancial_Report_2018_e.pdf"
doc = fitz.open(pdf_path)


relevant_sections = ["Integrity", "Compliance", "Sustainability", "Recovery", "Strategy", "Environmental Protection", "Governance"]


text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    page_text = page.get_text()
    if any(section in page_text for section in relevant_sections):
        text += page_text


def clean_text(text):
    # Remove unnecessary characters and patterns
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove standalone single letters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation but keep words
    
    
    text = text.lower()

    
    sentences = sent_tokenize(text)
    
    
    stop_words = set(stopwords.words('english'))
    
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha() and word not in stop_words]
        cleaned_sentence = ' '.join(words)
        cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences


cleaned_sentences = clean_text(text)

keywords = ["recovery", "strategy", "sustainability", "compliance", "integrity", "decarbonization", "governance", "responsibility", "mobility", "environmental"]
relevant_sentences = [sentence for sentence in cleaned_sentences if any(keyword in sentence for keyword in keywords)]


analyzer = SentimentIntensityAnalyzer()


keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

for sentence in relevant_sentences:
    for keyword in keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']


print("\nSentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[1]:


import fitz  # PyMuPDF
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('punkt')
nltk.download('stopwords')


pdf_path = r"C:\Users\LINCY\Downloads\Nonfinancial_Report_2019_en.pdf"
doc = fitz.open(pdf_path)


text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    page_text = page.get_text()
    text += page_text


def clean_and_extract_keywords(text):
    # Convert text to lowercase
    text = text.lower()
    
    
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
   
    stop_words = set(stopwords.words('english'))
    keywords_to_include = {"recovery", "strategy", "sustainability", "crisis", "compliance", "integrity", "decarbonization", "governance", "communication"}
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words and word in keywords_to_include]

    return filtered_words


filtered_keywords = clean_and_extract_keywords(text)


keyword_freq = Counter(filtered_keywords)


analyzer = SentimentIntensityAnalyzer()


def calculate_sentiment_for_keywords(text, keywords):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    
    keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}
    
    
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                sentiment = analyzer.polarity_scores(sentence)
                keyword_sentiments[keyword]['pos'] += sentiment['pos']
                keyword_sentiments[keyword]['neu'] += sentiment['neu']
                keyword_sentiments[keyword]['neg'] += sentiment['neg']
                keyword_sentiments[keyword]['compound'] += sentiment['compound']
                keyword_sentiments[keyword]['count'] += 1
    
    
    for keyword in keyword_sentiments:
        if keyword_sentiments[keyword]['count'] > 0:
            keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
            keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
            keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
            keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']
    
    return keyword_sentiments


keyword_sentiments = calculate_sentiment_for_keywords(text, keyword_freq.keys())


print("\nSentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[2]:


import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the PDF file
pdf_path = r"C:\Users\LINCY\Downloads\Nonfinancial_Report_2020_en.pdf"
doc = fitz.open(pdf_path)

# Extract text from all pages
text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    text += page.get_text()

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove standalone single letters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation but keep words
    text = text.lower()  # Lowercase the text
    sentences = sent_tokenize(text)  # Split text into sentences
    
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha() and word not in stop_words]
        cleaned_sentence = ' '.join(words)
        cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences

# Applying cleaning function to extracted text
cleaned_sentences = clean_text(text)

# Defining a list of keywords related to Volkswagen's recovery strategies
keywords = ["group", "volkswagen", "management", "sustainability", "business", 
            "compliance", "report", "employees", "vehicles", "emissions", 
            "mobility", "environmental", "integrity", "risks", "strategy"]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initializing dictionary to hold sentiment results for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

# Performing sentiment analysis on sentences containing each keyword
for sentence in cleaned_sentences:
    sentiment = analyzer.polarity_scores(sentence)
    for keyword in keywords:
        if keyword in sentence:
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculating average sentiment scores for each keyword
for keyword in keywords:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

# Printing the sentiment analysis results for each keyword
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")
    else:
        print(f"{keyword.capitalize()} - No relevant sentences found.")


# In[3]:


import fitz  # PyMuPDF
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensuring NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Loading the PDF file
pdf_path = r"C:\Users\LINCY\Downloads\letter released by volkswagen after dg scandal 2015.pdf"
doc = fitz.open(pdf_path)

# Extracting text from the PDF
text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    text += page.get_text()

# Preprocessing the text
stop_words = set(stopwords.words('english'))
stop_words.update(["volkswagen", "company", "group", "us", "also", "new", "year", "billion", "operating", "long", "special", "items", "set", "side", "one", "still", "much", "letter", "shareholders", "world"])  # Adding general non-informative words to stopwords
sentences = sent_tokenize(text.lower())
words_in_context = []

# Defining seed words related to recovery strategies
seed_words = ["recovery", "strategy", "trust", "compliance", "integrity", "rebuild", "crisis", "apology", "responsibility", "TOGETHER – Strategy 2025","New Partnerships and Participations", "Integrity and Sustainability", "Values and Culture"]

# Extracting sentences containing seed words and find relevant keywords
relevant_sentences = []
for sentence in sentences:
    if any(seed_word in sentence for seed_word in seed_words):
        relevant_sentences.append(sentence)
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha() and word not in stop_words]
        words_in_context.extend(words)

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Performing sentiment analysis on the filtered sentences
sentiment_results = []
for sentence in relevant_sentences:
    sentiment = analyzer.polarity_scores(sentence)
    sentiment_results.append(sentiment)

# Aggregating sentiment scores for the filtered sentences
overall_sentiment = {
    'pos': sum([sent['pos'] for sent in sentiment_results]) / len(sentiment_results),
    'neu': sum([sent['neu'] for sent in sentiment_results]) / len(sentiment_results),
    'neg': sum([sent['neg'] for sent in sentiment_results]) / len(sentiment_results),
    'compound': sum([sent['compound'] for sent in sentiment_results]) / len(sentiment_results)
}

print(f"\nOverall Sentiment Analysis Results:")
print(f"Positive: {overall_sentiment['pos']:.2f}")
print(f"Neutral: {overall_sentiment['neu']:.2f}")
print(f"Negative: {overall_sentiment['neg']:.2f}")
print(f"Compound: {overall_sentiment['compound']:.2f}")

# Performing keyword frequency analysis
keyword_freq = Counter(words_in_context)

# Get the most common keywords related to recovery strategies
common_keywords = keyword_freq.most_common(30)  # Adjust the number as needed

# Printing the results
if common_keywords:
    print("\nTop Keywords Related to Volkswagen's Recovery Strategies Post-Dieselgate:")
    for word, freq in common_keywords:
        print(f"{word}: {freq}")
else:
    print("No relevant keywords found.")



# In[4]:


import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string
import nltk
from collections import Counter

# Ensuring NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Loading the PDF file
pdf_path = r"C:\Users\LINCY\Downloads\2016.pdf"  # Update with your file path
doc = fitz.open(pdf_path)

# Extracting text from specific pages
text = ""
for page_num in range(89, 98):  # Pages are zero-indexed, so page 90 is index 89
    page = doc.load_page(page_num)
    text += page.get_text()

# Functioning to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
    text = re.sub(r'\s+', ' ', text)  
    return text

# Applying text cleaning
cleaned_text = clean_text(text)

# Spliting the text into sentences
sentences = sent_tokenize(cleaned_text)

# Tokenizing words in each sentence
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Defining relevant keywords using partial matches
relevant_keywords = ['recover', 'strateg', 'compliance', 'integrity', 'sustainab', 'govern', 'communic', 'crisis', 'trust', 'stakeholder', 'brand', 'management', 'reorgan', 'golden', 'monitor', 'train', 'optimiz', 'program', 'investig', 'cooperat', 'external', 'dismiss', 'environment', 'solution', 'clarification']

# Initializing dictionary to hold sentiment scores for each keyword
keyword_sentiments = {key: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for key in relevant_keywords}

# Analyzing sentiment for each sentence containing the keywords
for sentence, tokens in zip(sentences, tokenized_sentences):
    for key in relevant_keywords:
        if any(word.startswith(key) for word in tokens):
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[key]['pos'] += sentiment['pos']
            keyword_sentiments[key]['neu'] += sentiment['neu']
            keyword_sentiments[key]['neg'] += sentiment['neg']
            keyword_sentiments[key]['compound'] += sentiment['compound']
            keyword_sentiments[key]['count'] += 1

# Calculating average sentiment scores for each keyword
for key in keyword_sentiments:
    if keyword_sentiments[key]['count'] > 0:
        keyword_sentiments[key]['pos'] /= keyword_sentiments[key]['count']
        keyword_sentiments[key]['neu'] /= keyword_sentiments[key]['count']
        keyword_sentiments[key]['neg'] /= keyword_sentiments[key]['count']
        keyword_sentiments[key]['compound'] /= keyword_sentiments[key]['count']

# Printing the sentiment analysis results
print("Sentiment Analysis Results for Each Keyword:")
for key, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{key.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[5]:


import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Ensuring NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# URL of the webpage
url = "https://annualreport2015.volkswagenag.com/group-management-report/the-emissions-issue.html"

# Get the webpage content
response = requests.get(url)
web_content = response.text

# Parse the webpage content using BeautifulSoup
soup = BeautifulSoup(web_content, 'html.parser')

# Extract text from the webpage
text = soup.get_text(separator=' ')

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Apply text cleaning
cleaned_text = clean_text(text)

# Split the cleaned text into sentences
sentences = sent_tokenize(cleaned_text)

# Tokenize the text into words
words = word_tokenize(cleaned_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Define relevant keywords related to recovery strategies
relevant_keywords = [
    'technical', 'solutions', 'internal', 'inquiries', 'external', 'investigations',
    'cooperation', 'program', 'special', 'committee', 'remedial', 'measures',
    'organizational', 'procedural', 'improvements'
]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize dictionary to store sentiment scores for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in relevant_keywords}

# Analyzing sentiment for sentences containing the keywords
for sentence in sentences:
    for keyword in relevant_keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculate average sentiment scores for each keyword
for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

# Printing the sentiment analysis results
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[6]:


import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import string

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define the URL of the webpage to scrape
url = "https://annualreport2016.volkswagenag.com/group-management-report/goals-and-strategies.html"

# Sending a GET request to the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all the text from the webpage
    text = soup.get_text()

    # Function to clean and preprocess text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    # Apply text cleaning
    cleaned_text = clean_text(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)

    # Keywords to search for
    relevant_keywords = [
        'together', 'strategy', 'sustainable', 'mobility', 'technological', 
        'leadership', 'customer', 'oriented', 'integrity', 'reliability', 
        'efficiency', 'profitability', 'code', 'collaboration'
    ]

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Initialize dictionary to store sentiment scores for each keyword
    keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in relevant_keywords}

    # Analyze sentiment for sentences containing the keywords
    for sentence in sentences:
        for keyword in relevant_keywords:
            if keyword in sentence:
                sentiment = analyzer.polarity_scores(sentence)
                keyword_sentiments[keyword]['pos'] += sentiment['pos']
                keyword_sentiments[keyword]['neu'] += sentiment['neu']
                keyword_sentiments[keyword]['neg'] += sentiment['neg']
                keyword_sentiments[keyword]['compound'] += sentiment['compound']
                keyword_sentiments[keyword]['count'] += 1

    # Calculate average sentiment scores for each keyword
    for keyword in keyword_sentiments:
        if keyword_sentiments[keyword]['count'] > 0:
            keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
            keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
            keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
            keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

    # Print the sentiment analysis results
    print("Sentiment Analysis Results for Each Keyword:")
    for keyword, scores in keyword_sentiments.items():
        if scores['count'] > 0:
            print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")

else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# In[1]:


import fitz  # PyMuPDF
import re
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Load the PDF file
pdf_path = r"C:\Users\LINCY\Downloads\360_Okt_2019_WOB_EN.pdf"  # Update with your file path
doc = fitz.open(pdf_path)

# Extract text from the document
text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    text += page.get_text()

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Apply text cleaning
cleaned_text = clean_text(text)

# Tokenize the text into sentences
sentences = sent_tokenize(cleaned_text)

# Define relevant keywords related to recovery strategies
recovery_keywords = [
    "compliance", "monitor", "ethics", "integrity", "legal affairs", "settlement",
    "certification", "transparency", "trust", "accountability", "restructuring",
    "recovery", "strategy", "technical solutions", "organizational changes", 
    "remedial measures", "communication", "stakeholder engagement", "credibility",
    "internal inquiries", "external investigations", "cooperation program",
    "process improvement", "remediation", "audit", "risk management", 
    "leadership", "responsibility", "training", "employee engagement"
]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initializing a dictionary to store sentiment scores for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in recovery_keywords}

# Analyzing sentiment for sentences containing the keywords
for sentence in sentences:
    for keyword in recovery_keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculating average sentiment scores for each keyword
for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

# Printing the sentiment analysis results
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[2]:


import requests
from bs4 import BeautifulSoup
from collections import Counter
import re
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensuring NLTK data is downloaded
nltk.download('punkt')

# URL of the webpage
url = "https://www.justice.gov/opa/pr/volkswagen-ag-agrees-plead-guilty-and-pay-43-billion-criminal-and-civil-penalties-six"

# Send a GET request to the webpage
response = requests.get(url)

# Parsing the webpage content
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting all text from the webpage
text = soup.get_text()

# Functioning to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Applying text cleaning
cleaned_text = clean_text(text)

# Tokenizing the text into sentences
sentences = sent_tokenize(cleaned_text)

# Expanded keywords related to recovery strategies
keywords = [
    "guilty plea", "criminal" , "civil penalties", "corporate", "compliance monitor", 
    "injunctive relief", "settlement", "remedial measures", "reforms", 
    "governance changes", "audit", "oversight", "legal settlements", 
    "compliance programs", "reorganization", "transparency", "ethics training", 
    "recovery plan", "risk management", "policy changes", "investigations", 
    "integrity program", "compliance measures", "corrective actions"
]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initializing a dictionary to store sentiment scores for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

# Analyzing sentiment for sentences containing the keywords
for sentence in sentences:
    for keyword in keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculating average sentiment scores for each keyword
for keyword in keyword_sentiments:
    if keyword_sentiments[keyword]['count'] > 0:
        keyword_sentiments[keyword]['pos'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neu'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['neg'] /= keyword_sentiments[keyword]['count']
        keyword_sentiments[keyword]['compound'] /= keyword_sentiments[keyword]['count']

# Printing the sentiment analysis results
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[ ]:




