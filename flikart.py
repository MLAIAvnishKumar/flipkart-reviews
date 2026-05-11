import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/flipkart_reviews.csv")

# Preview the data
print(data.head())


import nltk
import re
import string
from nltk.corpus import stopwords

# Download the list of common filler words (like 'the', 'a', 'is')
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply the cleaning function to your 'Review' column
data["Review"] = data["Review"].apply(clean)

data["Rating"]

data['Review']

print(data.isnull().sum())
import plotly.express as px

ratings = data["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values

# Remove "data" from the first position here:
figure = px.pie(values=quantity, 
                names=numbers, 
                hole=0.5)
figure.show()


# Combine all the cleaned reviews into one long string
text = " ".join(i for i in data.Review)

# Set up stopwords and generate the word cloud
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




