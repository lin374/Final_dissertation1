#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Loading the data
file_path = r"C:\Users\LINCY\OneDrive\Tài liệu\sentiment analysis for all the articles.xlsx"
data = pd.read_excel(file_path)

# Aggregating data if needed (ensuring each keyword has a unique entry per report/article)
aggregated_data = data.groupby(['Keyword', 'Reports/Articles']).mean().reset_index()

# Pivoting the data for the heatmap
heatmap_data = aggregated_data.pivot(index='Keyword', columns='Reports/Articles', values='Compound')

# Handling NaN values by filling with a neutral value (e.g., 0)
heatmap_data = heatmap_data.fillna(0)

# Ensuring all values are finite
heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0)

# Checking for any remaining NaN or infinite values
if not np.isfinite(heatmap_data.values).all():
    print("There are still infinite or NaN values in the data.")
    heatmap_data = heatmap_data.applymap(lambda x: 0 if not np.isfinite(x) else x)

# Creating the heatmap without clustering if the clustering fails
try:
    # Reorder the keywords based on their sentiment profiles using hierarchical clustering
    clustered_data = heatmap_data.loc[sns.clustermap(heatmap_data, cmap="coolwarm", standard_scale=1, figsize=(10, 10)).dendrogram_row.reordered_ind]
except ValueError:
    print("Clustering failed, proceeding without clustering.")
    clustered_data = heatmap_data

# Creating the heatmap with a more contrasting color palette and annotations
plt.figure(figsize=(14, 10))
sns.heatmap(clustered_data, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Compound Sentiment Score'})
plt.title("Heatmap of Sentiment Analysis per Keyword Across Articles")
plt.xlabel("Report/Article")
plt.ylabel("Keyword")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  # Keep the y-axis labels horizontal for better readability
plt.tight_layout()

# Saveing the heatmap
plt.savefig("clear_concise_heatmap.png", dpi=300)

# Showing the heatmap
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data from your Excel file
data_new = pd.read_excel(r"C:\Users\LINCY\OneDrive\Tài liệu\sentiment analysis for all the articles.xlsx", sheet_name='Sheet1')

# Dropping rows with NaN values in the 'Keyword' column, since those rows don't contribute to the keyword analysis
data_new = data_new.dropna(subset=['Keyword'])

# Grouping by 'Keyword' and sum the sentiment values
sentiment_grouped = data_new.groupby('Keyword')[['Positive', 'Neutral', 'Negative']].sum()

# Normalizing the sentiment data to get proportions
sentiment_grouped_normalized = sentiment_grouped.div(sentiment_grouped.sum(axis=1), axis=0)

# Simplifying by focusing on the top keywords by overall sentiment (sum of proportions)
top_keywords = sentiment_grouped_normalized.sum(axis=1).sort_values(ascending=False).head(20).index  # Changed to 20
sentiment_grouped_top = sentiment_grouped_normalized.loc[top_keywords]

# Create a more visually appealing stacked bar chart using seaborn
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Plotting the stacked bar chart
sentiment_grouped_top.plot(kind='bar', stacked=True, figsize=(12, 8), color=['#2ecc71', '#f1c40f', '#e74c3c'])

# Enhancing the plot
plt.title('Top 20 Keywords Sentiment Distribution', fontsize=16)  # Updated title
plt.xlabel('Keyword', fontsize=14)
plt.ylabel('Sentiment Proportion', fontsize=14)
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
sns.despine(left=True, bottom=True)

# Adding annotations for better insights
for i in range(sentiment_grouped_top.shape[0]):
    for j in range(sentiment_grouped_top.shape[1]):
        plt.text(i, 
                 sentiment_grouped_top.iloc[i, :j+1].sum() - sentiment_grouped_top.iloc[i, j]/2, 
                 f'{sentiment_grouped_top.iloc[i, j]:.2f}', 
                 ha='center', va='center', color='black', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path =  r"C:\Users\LINCY\OneDrive\Tài liệu\according to dates.xlsx" 
data_with_dates = pd.read_excel(file_path)

# Cleaning the data by dropping rows where 'Date/Year' is NaN
data_with_dates_clean = data_with_dates.dropna(subset=['Date/Year'])

# Ensuring the 'Date/Year' column is in datetime format (assuming it's a year for simplicity)
data_with_dates_clean['Date/Year'] = pd.to_datetime(data_with_dates_clean['Date/Year'], format='%Y')

# Grouping by 'Date/Year' and calculate the mean sentiment proportions
sentiment_trend = data_with_dates_clean.groupby(data_with_dates_clean['Date/Year'].dt.year)[['Positive', 'Neutral', 'Negative']].mean()

# Enhanced Plotting with Seaborn for better aesthetics
sns.set(style="whitegrid")  # Use a white grid background for clarity

plt.figure(figsize=(12, 8))

# Plotting the sentiment trends
plt.plot(sentiment_trend.index, sentiment_trend['Positive'], label='Positive', marker='o', color='#2ecc71', linewidth=2.5)
plt.plot(sentiment_trend.index, sentiment_trend['Neutral'], label='Neutral', marker='o', color='#f1c40f', linewidth=2.5)
plt.plot(sentiment_trend.index, sentiment_trend['Negative'], label='Negative', marker='o', color='#e74c3c', linewidth=2.5)

# Adding a trend line (optional smoothing example with moving average)
plt.plot(sentiment_trend.index, sentiment_trend['Positive'].rolling(window=2).mean(), color='#27ae60', linestyle='--', linewidth=1.5)
plt.plot(sentiment_trend.index, sentiment_trend['Neutral'].rolling(window=2).mean(), color='#f39c12', linestyle='--', linewidth=1.5)
plt.plot(sentiment_trend.index, sentiment_trend['Negative'].rolling(window=2).mean(), color='#c0392b', linestyle='--', linewidth=1.5)

# Adding annotations for significant points
for i in range(len(sentiment_trend)):
    plt.text(sentiment_trend.index[i], sentiment_trend['Positive'].iloc[i] + 0.02, f'{sentiment_trend["Positive"].iloc[i]:.2f}', 
             color='#2ecc71', ha='center')
    plt.text(sentiment_trend.index[i], sentiment_trend['Negative'].iloc[i] - 0.03, f'{sentiment_trend["Negative"].iloc[i]:.2f}', 
             color='#e74c3c', ha='center')
    plt.text(sentiment_trend.index[i], sentiment_trend['Neutral'].iloc[i] + 0.02, f'{sentiment_trend["Neutral"].iloc[i]:.2f}', 
             color='#f1c40f', ha='center')

# Enhancing the plot
plt.title('Sentiment Trends Over Time', fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Sentiment Proportion', fontsize=14)
plt.legend(title='Sentiment', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # Add a light grid with dashed lines for better readability

# Adding context with a subtitle
plt.suptitle('Tracking Public Perception of Volkswagen Post-Dieselgate', fontsize=14, color='gray')

# Display the plot
plt.tight_layout()
plt.show()


# In[7]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_excel(r"C:\Users\LINCY\OneDrive\Tài liệu\arranged by years (volkswagen reports.xlsx")

# Generating the word cloud based on keyword frequency
text = " ".join(keyword for keyword in df['Keyword'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Displaying the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Keyword Frequency Analysis')
plt.show()


# In[8]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Load your data from the Excel file
data = pd.read_excel(r"C:\Users\LINCY\OneDrive\Tài liệu\sentiment analysis for all the articles.xlsx")  # Update with your actual file path

# Calculate keyword frequency if 'Frequency' column doesn't exist
keyword_list = data['Keyword'].tolist()
keyword_freq_dict = dict(Counter(keyword_list))

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_freq_dict)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis
plt.title("Word Cloud for Keyword Frequency Analysis", fontsize=16)
plt.show()


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r"C:\Users\LINCY\OneDrive\Tài liệu\volkswagen data from website.xlsx"
volkswagen_data = pd.read_excel(file_path)

# Clean the data by dropping rows where 'Keyword' is NaN, as these don't contribute to the keyword analysis
volkswagen_data_clean = volkswagen_data.dropna(subset=['Keyword'])

# Group by 'Keyword' and sum the sentiment values
keyword_sentiment = volkswagen_data_clean.groupby('Keyword')[['Positive', 'Neutral', 'Negative']].sum()

# Calculate the total frequency of each keyword
keyword_sentiment['Total'] = keyword_sentiment.sum(axis=1)

# Sort by total frequency to focus on the most mentioned keywords
keyword_sentiment_sorted = keyword_sentiment.sort_values(by='Total', ascending=False).head(20)

# Normalize the sentiment values to get proportions
keyword_sentiment_normalized = keyword_sentiment_sorted.div(keyword_sentiment_sorted['Total'], axis=0)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(keyword_sentiment_normalized[['Positive', 'Neutral', 'Negative']], annot=True, cmap='coolwarm', linewidths=.5)

# Enhancing the plot
plt.title('Keyword Sentiment Heatmap (Top 20 Keywords)', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Keyword', fontsize=14)

# Display the plot
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from your Excel file
data_new = pd.read_excel(r"C:\Users\LINCY\OneDrive\Tài liệu\arranged by years (volkswagen reports.xlsx", sheet_name='Sheet1')

# Dropping rows with NaN values in the 'Keyword' column, since those rows don't contribute to the keyword analysis
data_new = data_new.dropna(subset=['Keyword'])

# Group by 'Keyword' and sum the sentiment values
sentiment_grouped = data_new.groupby('Keyword')[['Positive', 'Neutral', 'Negative']].sum()

# Normalize the sentiment data to get proportions
sentiment_grouped_normalized = sentiment_grouped.div(sentiment_grouped.sum(axis=1), axis=0)

# Simplify by focusing on the top keywords by overall sentiment (sum of proportions)
top_keywords = sentiment_grouped_normalized.sum(axis=1).sort_values(ascending=False).head(20).index  # Changed to 20
sentiment_grouped_top = sentiment_grouped_normalized.loc[top_keywords]

# Create a more visually appealing stacked bar chart using seaborn
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Plotting the stacked bar chart
sentiment_grouped_top.plot(kind='bar', stacked=True, figsize=(12, 8), color=['#2ecc71', '#f1c40f', '#e74c3c'])

# Enhancing the plot
plt.title('Top 20 Keywords Sentiment Distribution', fontsize=16)  # Updated title
plt.xlabel('Keyword', fontsize=14)
plt.ylabel('Sentiment Proportion', fontsize=14)
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
sns.despine(left=True, bottom=True)

# Adding annotations for better insights
for i in range(sentiment_grouped_top.shape[0]):
    for j in range(sentiment_grouped_top.shape[1]):
        plt.text(i, 
                 sentiment_grouped_top.iloc[i, :j+1].sum() - sentiment_grouped_top.iloc[i, j]/2, 
                 f'{sentiment_grouped_top.iloc[i, j]:.2f}', 
                 ha='center', va='center', color='black', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()



# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_excel(r"C:\Users\LINCY\OneDrive\Tài liệu\arranged by years (volkswagen reports.xlsx")

# Ensure the 'Dates/Year' column is a string
data['Dates/Year'] = data['Dates/Year'].astype(str)

# Extract the year from the 'Dates/Year' column
data['Year'] = data['Dates/Year'].str.extract(r'(\d{4})')

# Droping rows where year could not be extracted
data = data.dropna(subset=['Year'])

# Converting the 'Year' column to integer type
data['Year'] = data['Year'].astype(int)

# Converting sentiment columns to numeric, forcing errors to NaN (which will be ignored)
sentiment_columns = ['Positive', 'Neutral', 'Negative', 'Compound']
for col in sentiment_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Grouping by year and calculate the mean sentiment scores
yearly_sentiment = data.groupby('Year')[sentiment_columns].mean().reset_index()

# Set the figure size for a more detailed view
plt.figure(figsize=(14, 8))

# Plotting sentiment trends over time
sns.lineplot(data=yearly_sentiment, x='Year', y='Positive', marker='o', label='Positive')
sns.lineplot(data=yearly_sentiment, x='Year', y='Neutral', marker='o', label='Neutral')
sns.lineplot(data=yearly_sentiment, x='Year', y='Negative', marker='o', label='Negative')
sns.lineplot(data=yearly_sentiment, x='Year', y='Compound', marker='o', label='Compound')

# Adding titles and labels
plt.title('Sentiment Trends Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Sentiment Score', fontsize=14)

# Add annotations to explain key points
for i, row in yearly_sentiment.iterrows():
    plt.text(row['Year'], row['Positive'] + 0.02, f"{row['Positive']:.2f}", color='blue', ha='center')
    plt.text(row['Year'], row['Neutral'] + 0.02, f"{row['Neutral']:.2f}", color='orange', ha='center')
    plt.text(row['Year'], row['Negative'] + 0.02, f"{row['Negative']:.2f}", color='green', ha='center')
    plt.text(row['Year'], row['Compound'] + 0.02, f"{row['Compound']:.2f}", color='red', ha='center')

# Show legend
plt.legend(title='Sentiment')

# Show grid for better readability
plt.grid(True)

# Save the plot
plt.savefig("sentiment_trends_over_time.png")

# Display the plot
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import pandas as pd

# Manually entering the data from the images
data = {
    'Year': list(range(2006, 2024)),
    'Operating Profit (in million euros)': [2009, 6151, 6333, 1855, 7141, 11271, 11498, 11671, 12697, -4069, 7103, 13818, 13920, 16960, 9675, 19275, 22109, 22576],
    'Sales Revenue (in billion euros)': [104.88, 108.90, 113.81, 105.19, 126.88, 159.34, 192.68, 197.01, 202.46, 213.29, 217.27, 229.55, 235.85, 252.63, 222.88, 250.20, 279.05, 322.28]
}

# Converting to DataFrame
df = pd.DataFrame(data)

# Rescale the Sales Revenue to the same scale as Operating Profit (multiply by 1000 to convert from billion to million)
df['Rescaled Sales Revenue (in million euros)'] = df['Sales Revenue (in billion euros)'] * 1000

# Plotting the financial trends with annotations for key events
plt.figure(figsize=(16, 9))  # Increase the figure size

# Plotting Operating Profit
plt.plot(df['Year'], df['Operating Profit (in million euros)'], label='Operating Profit (in million euros)', color='green', marker='o')

# Plotting Sales Revenue (rescaled)
plt.plot(df['Year'], df['Rescaled Sales Revenue (in million euros)'], label='Sales Revenue (rescaled to million euros)', color='blue', marker='o')

# Highlighting key events with annotations
events = {
    2015: 'Dieselgate Scandal Announced',
    2016: 'First Legal Settlements',
    2018: 'EV Strategy Launched',
    2020: 'COVID-19 Impact',
    2021: 'New EV Models Released'
}

# Adding annotations to the plot with better positioning
for year, event in events.items():
    y_value = df.loc[df['Year'] == year, 'Operating Profit (in million euros)'].values[0]
    plt.annotate(event, 
                 xy=(year, y_value), 
                 xytext=(0, 80),  # Adjust text position
                 textcoords='offset points',
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=10, 
                 ha='center')

# Adding titles and labels
plt.title('Volkswagen Group Financial Trends with Key Events (2006-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Value (in million euros)', fontsize=14)

# Adding a legend
plt.legend()

# Adjusting the layout manually to avoid clipping
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Display the plot
plt.grid(True)
plt.show()


# In[ ]:




