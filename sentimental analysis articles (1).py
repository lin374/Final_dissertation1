#!/usr/bin/env python
# coding: utf-8

# In[1]:


year2015
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensuring NLTK data is downloaded
nltk.download('punkt')

# URL of the webpage
url = "https://hub.jhu.edu/2015/12/07/volkswagen-scandal-explained-sylvia-long-tolbert/"

# Getting the webpage content
response = requests.get(url)
web_content = response.text

# Parsing the webpage content using BeautifulSoup
soup = BeautifulSoup(web_content, 'html.parser')

# Extracting the text from the webpage
text = soup.get_text(separator=' ')

# Tokenizing the text into sentences
sentences = sent_tokenize(text)

# Define the relevant keywords for sentiment analysis
keywords = ["recover", "scandal", "customer", "legal", "strateg", "settlement", "ethic", "trust", "rebuild"]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Performing sentiment analysis on sentences containing the keywords
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

for sentence in sentences:
    for keyword in keywords:
        if keyword in sentence.lower():
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
    else:
        print(f"{keyword.capitalize()} - No relevant sentences found.")


# In[5]:


year2017
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk

# Ensuring NLTK data is downloaded
nltk.download('punkt')

# URL of the webpage
url = "https://knowledge.wharton.upenn.edu/podcast/knowledge-at-wharton-podcast/can-volkswagen-move-beyond-its-diesel-emissions-scandal/"

# Sending a GET request to the webpage
response = requests.get(url)

# Parsing the webpage content
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting all text from the webpage
text = soup.get_text()

# Functioning to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Applying text cleaning
cleaned_text = clean_text(text)

# Spliting text into sentences
sentences = sent_tokenize(cleaned_text)

# Defining relevant keywords
recovery_keywords = ["leadership", "settlement", "trust", "ethics", "responsibility", "crisis", "compliance", "strategy"]

# Initializing VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initializing a dictionary to store sentiment scores
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in recovery_keywords}

# Perform sentiment analysis on sentences containing the keywords
for sentence in sentences:
    for keyword in recovery_keywords:
        if keyword in sentence:
            sentiment = analyzer.polarity_scores(sentence)
            keyword_sentiments[keyword]['pos'] += sentiment['pos']
            keyword_sentiments[keyword]['neu'] += sentiment['neu']
            keyword_sentiments[keyword]['neg'] += sentiment['neg']
            keyword_sentiments[keyword]['compound'] += sentiment['compound']
            keyword_sentiments[keyword]['count'] += 1

# Calculatinhgaverage sentiment scores for each keyword
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


year2023
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from collections import Counter
import nltk

# Ensure NLTK data is downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# URL of the webpage
url = "https://changemanagementinsight.com/volkswagen-crisis-management-case-study/#:~:text=The%20lessons%20learned%20from%20this,are%20vital%20for%20rebuilding%20trust."

# Sending a GET request to the webpage
response = requests.get(url)

# Parse the webpage content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting all text from the webpage
text = soup.get_text(separator=' ')

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Applying text cleaning
cleaned_text = clean_text(text)

# Tokenizing the text into sentences
sentences = nltk.sent_tokenize(cleaned_text)

# Define keywords to identify recovery strategies
recovery_keywords = [
    "management", "trust", "response", "communication", "leadership", 
    "integrity", "public", "responsibility", "ethics", "credibility",
    "customer", "legal", "recovery", "measures", "transparency", 
    "accountability", "reaction", "technical", "strategy", "remedial",
    "organizational", "changes", "compliance", "risk", "stakeholder"
]

# Initializing a dictionary to store sentiment results for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in recovery_keywords}

# Perform sentiment analysis for sentences containing the keywords
for sentence in sentences:
    sentiment = analyzer.polarity_scores(sentence)
    for keyword in recovery_keywords:
        if keyword in sentence:
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

# Printing the sentiment analysis results for each keyword
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[7]:


import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import re
import nltk

# Ensure NLTK data is downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')

# Define the URL of the webpage to scrape
url = "https://hbr.org/2016/09/the-scandal-effect"

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
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    # Apply text cleaning
    cleaned_text = clean_text(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)

    # Define relevant keywords related to recovery strategies
    relevant_keywords = [
        "transparency", "strategy", "organizational", "inquiries", 
        "investigations", "risk", "leadership", "customer", 
        "reputation", "management", "sustainability", "government"
    ]

    # Initializing VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Perform sentiment analysis for sentences containing the keywords
    sentiment_results = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in relevant_keywords}

    for sentence in sentences:
        for keyword in relevant_keywords:
            if keyword in sentence:
                sentiment = analyzer.polarity_scores(sentence)
                sentiment_results[keyword]['pos'] += sentiment['pos']
                sentiment_results[keyword]['neu'] += sentiment['neu']
                sentiment_results[keyword]['neg'] += sentiment['neg']
                sentiment_results[keyword]['compound'] += sentiment['compound']
                sentiment_results[keyword]['count'] += 1

    # Calculate average sentiment scores for each keyword
    for keyword, scores in sentiment_results.items():
        if scores['count'] > 0:
            scores['pos'] /= scores['count']
            scores['neu'] /= scores['count']
            scores['neg'] /= scores['count']
            scores['compound'] /= scores['count']

    # Print the sentiment analysis results
    print("Sentiment Analysis Results for Each Keyword:")
    for keyword, scores in sentiment_results.items():
        if scores['count'] > 0:
            print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")
        else:
            print(f"{keyword.capitalize()} - No relevant sentences found.")

else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# In[9]:


import requests
from bs4 import BeautifulSoup
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# URL of the webpage
url = "https://www.forbes.com/sites/greatspeculations/2015/10/06/volkswagen-emissions-scandal-what-we-can-learn-from-history/"

# Send a GET request to the webpage
response = requests.get(url)

# Parse the webpage content
soup = BeautifulSoup(response.content, 'html.parser')

# Extract all text from the webpage
text = soup.get_text()

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

# Define specific keywords related to recovery strategies
recovery_keywords = [
    "recovery", "strategy", "compliance", "ethics", "integrity", "trust", "transparency",
    "settlement", "leadership", "accountability", "responsibility", "communication",
    "stakeholder", "remediation", "governance", "credibility", "risk management",
    "technical solutions", "reputation", "public relations", "organizational changes",
    "remedial measures", "legal", "monitor", "training", "employee engagement"
]

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Dictionary to store sentiment results for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in recovery_keywords}

# Perform sentiment analysis for each keyword
for sentence in sentences:
    sentiment = analyzer.polarity_scores(sentence)
    for keyword in recovery_keywords:
        if keyword in sentence:
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
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[11]:


import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define the URL of the webpage to scrape
url = "https://www.forbes.com/sites/georgkell/2022/12/05/from-emissions-cheater-to-climate-leader-vws-journey-from-dieselgate-to-embracing-e-mobility/"

# Send a GET request to the webpage
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
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    # Apply text cleaning
    cleaned_text = clean_text(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)

    # Define relevant keywords related to recovery strategies
    relevant_keywords = [
        'sustainability', 'settlement', 'leadership', 'strategy', 'management',
        'reputation', 'governance', 'engagement', 'integrity', 'compliance', 
        'legal', 'transformation', 'innovation', 'recovery', 'trust', 'cooperation'
    ]

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Perform sentiment analysis for sentences containing the keywords
    keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in relevant_keywords}

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


# In[12]:


import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('vader_lexicon')

# URL of the webpage
url = "https://www.wired.com/2015/10/vw-plans-to-recover-from-its-scandal-by-going-electric/"

# Send a GET request to the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the webpage content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all text from the webpage
    text = soup.get_text()

    # Function to clean and preprocess text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'[\s]+', ' ', text).strip()  # Replace multiple spaces with a single space and trim
        return text

    # Apply text cleaning
    cleaned_text = clean_text(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)

    # Define keywords for analysis
    keywords = ["scandal", "electric", "strategy", "clean", "emissions", "management", "technology", "future", "environmental", "transformation"]

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
    print("Sentiment Analysis Results for Each Keyword:")
    for keyword, scores in keyword_sentiments.items():
        if scores['count'] > 0:
            print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")
        else:
            print(f"{keyword.capitalize()} - No relevant sentences found.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# In[13]:


import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensuring NLTK data is downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Functioning to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Define your extracted keywords
keywords = ["settlement", "emissions", "scandal", "compliance", "trust", "fine", "crisis"]

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Example text (you would replace this with your actual text data)
# Assuming you have text data extracted using the method in the previous steps
text_data = """
Volkswagen has agreed to a settlement to resolve the emissions scandal. 
The company faced significant fines and compliance measures to regain trust.
The crisis has severely impacted their reputation.
"""

# Clean the text
cleaned_text = clean_text(text_data)

# Tokenize text into sentences
sentences = sent_tokenize(cleaned_text)

# Initialize a dictionary to store sentiment scores for each keyword
keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in keywords}

# Analyze sentiment for sentences containing each keyword
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

# Print sentiment analysis results for each keyword
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    if scores['count'] > 0:
        print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")
    else:
        print(f"{keyword.capitalize()} - No relevant sentences found.")


# In[14]:


import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# URL of the webpage
url = "https://www.npr.org/2018/04/24/605014988/after-diesel-scandal-vw-turns-to-new-leadership-and-electric-cars"

# Sending a GET request to the webpage
response = requests.get(url)

# Checking if the request was successful
if response.status_code == 200:
    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the text from the webpage
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

    # Tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)

    # Define relevant keywords related to the Dieselgate scandal and recovery strategies
    relevant_keywords = [
        'scandal', 'leadership', 'electric', 'management', 'emissions', 'settlement', 'technology'
    ]

    # Filtering sentences that contain the keywords
    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence for keyword in relevant_keywords)]

    # Initializing VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Performing sentiment analysis for each keyword
    keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in relevant_keywords}

    for sentence in relevant_sentences:
        sentiment = analyzer.polarity_scores(sentence)
        for keyword in relevant_keywords:
            if keyword in sentence:
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


# In[15]:


import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# URL of the webpage
url = "https://www.cleanenergywire.org/factsheets/dieselgate-forces-vw-embrace-green-mobility"

# Send a GET request to the webpage
response = requests.get(url)

# Checking if the request was successful
if response.status_code == 200:
    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the text from the webpage
    text = soup.get_text(separator=' ')

    # Functioning to clean and preprocess text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    # Applying text cleaning
    cleaned_text = clean_text(text)

    # Tokenizing the text into sentences
    sentences = sent_tokenize(cleaned_text)

    # Defining relevant keywords related to Dieselgate scandal and recovery strategies
    relevant_keywords = [
        "green", "mobility", "transformation", "emissions", "scandal", 
        "dieselgate", "electric", "sustainability", "strategy", 
        "technology", "management"
    ]

    # Initializing VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Initializing a dictionary to store sentiment scores for each keyword
    keyword_sentiments = {keyword: {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0, 'count': 0} for keyword in relevant_keywords}

    # Performing sentiment analysis on sentences containing the keywords
    for sentence in sentences:
        sentiment = analyzer.polarity_scores(sentence)
        for keyword in relevant_keywords:
            if keyword in sentence:
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
    print("\nSentiment Analysis Results for Each Keyword:")
    for keyword, scores in keyword_sentiments.items():
        if scores['count'] > 0:
            print(f"{keyword.capitalize()} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")
        else:
            print(f"{keyword.capitalize()} - No relevant sentences found.")

else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# In[16]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define the relevant keywords and their frequency
keywords_with_frequency = {
    "Electric": 19,
    "Dieselgate": 7,
    "Emissions": 3,
    "Cheating": 1,
    "Scandal": 1,
    "Fines": 2,
    "Reputation": 1,
    "Innovation": 1,
    "Strategy": 1,
    "Technology": 1,
    "Deception": 1
}

# Function to perform sentiment analysis on each keyword
def analyze_sentiment(keyword):
    sentiment = analyzer.polarity_scores(keyword)
    return sentiment

# Analyzing sentiment for each keyword
keyword_sentiments = {}
for keyword in keywords_with_frequency:
    sentiment = analyze_sentiment(keyword)
    keyword_sentiments[keyword] = sentiment

# Printing sentiment analysis results
print("Sentiment Analysis Results for Each Keyword:")
for keyword, sentiment in keyword_sentiments.items():
    print(f"{keyword} - Positive: {sentiment['pos']:.2f}, Neutral: {sentiment['neu']:.2f}, Negative: {sentiment['neg']:.2f}, Compound: {sentiment['compound']:.2f}")


# In[1]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# List of relevant keywords and their counts
keywords = {
    "Electric": 8,
    "Sustainability": 7,
    "Emissions": 2,
    "Dieselgate": 4,
    "Scandal": 3,
    "Management": 2,
    "Strategy": 6,
    "Compliance": 2,
    "Leadership": 2
}

# Example sentences for each keyword (in practice, you would extract these from your text)
# Here, I'm just using placeholders for demonstration.
sentences = {
    "Electric": ["Volkswagen is focusing on electric vehicles to rebuild its image.", 
                 "The electric strategy is seen as a pivotal change."],
    "Sustainability": ["Volkswagen's commitment to sustainability is questioned after Dieselgate.", 
                       "Sustainability has become a central focus post-scandal."],
    "Emissions": ["Emissions cheating scandal has hurt Volkswagen's reputation.", 
                  "The company has vowed to cut emissions moving forward."],
    "Dieselgate": ["Dieselgate remains a stain on Volkswagen's history.", 
                   "Post-Dieselgate, Volkswagen has made several reforms."],
    "Scandal": ["The scandal has impacted Volkswagen's global trust.", 
                "The company is working to recover from the emissions scandal."],
    "Management": ["Volkswagen's management is under scrutiny post-Dieselgate.", 
                   "Management changes are part of the recovery strategy."],
    "Strategy": ["Volkswagen's new strategy focuses on electrification.", 
                 "Strategy shifts are aimed at restoring trust."],
    "Compliance": ["Compliance with environmental standards is now a priority.", 
                   "Volkswagen's compliance programs have been overhauled."],
    "Leadership": ["Leadership changes were made to steer the company post-crisis.", 
                   "New leadership is focusing on transparency and accountability."]
}

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis for each keyword
keyword_sentiments = {}
for keyword, sentence_list in sentences.items():
    sentiment_scores = {"pos": 0, "neu": 0, "neg": 0, "compound": 0, "count": 0}
    
    for sentence in sentence_list:
        sentiment = analyzer.polarity_scores(sentence)
        sentiment_scores["pos"] += sentiment["pos"]
        sentiment_scores["neu"] += sentiment["neu"]
        sentiment_scores["neg"] += sentiment["neg"]
        sentiment_scores["compound"] += sentiment["compound"]
        sentiment_scores["count"] += 1
    
    # Calculate average sentiment for the keyword
    if sentiment_scores["count"] > 0:
        sentiment_scores["pos"] /= sentiment_scores["count"]
        sentiment_scores["neu"] /= sentiment_scores["count"]
        sentiment_scores["neg"] /= sentiment_scores["count"]
        sentiment_scores["compound"] /= sentiment_scores["count"]
    
    keyword_sentiments[keyword] = sentiment_scores

# Print the sentiment analysis results for each keyword
print("Sentiment Analysis Results for Each Keyword:")
for keyword, scores in keyword_sentiments.items():
    print(f"{keyword} - Positive: {scores['pos']:.2f}, Neutral: {scores['neu']:.2f}, Negative: {scores['neg']:.2f}, Compound: {scores['compound']:.2f}")


# In[ ]:




