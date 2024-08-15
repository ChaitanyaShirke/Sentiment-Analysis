from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import networkx as nx
from wordcloud import WordCloud
from collections import Counter


app = Flask(__name__)

# Load the trained model and vectorizer
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_review(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [word for word in tokens if word not in string.punctuation]
    return ' '.join(tokens)

def determine_sentiment(text):
    blob = TextBlob(text)
    return 1 if blob.sentiment.polarity > 0 else 0

def plot_pie_chart(df):
    sentiment_counts = df['sentiment_textblob'].value_counts()
    labels = ['Negative', 'Positive']
    sizes = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]

    plt.figure(figsize=(10, 8), facecolor='#f4f4f9')
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#FF6666', '#66FF66'])
    plt.title('Sentiment Distribution', color='orange')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#f4f4f9')
    img.seek(0)
    pie_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return pie_chart_url

def plot_bar_chart(df):
    sentiment_counts = df['sentiment_textblob'].value_counts()
    labels = ['Negative', 'Positive']
    sizes = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]

    plt.figure(figsize=(10, 8), facecolor='#f4f4f9')
    plt.bar(labels, sizes, color=['#FF6666', '#66FF66'])
    plt.title('Sentiment Distribution', color='orange')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#f4f4f9')
    img.seek(0)
    bar_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return bar_chart_url

def plot_sentiment_trends(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'sentiment_textblob'])
    df['sentiment_textblob'] = pd.to_numeric(df['sentiment_textblob'], errors='coerce')
    df = df.dropna(subset=['sentiment_textblob'])
    df.set_index('date', inplace=True)
    
    categories = df['category'].unique()
    plt.figure(figsize=(14, 8), facecolor='#f4f4f9')
    for category in categories:
        category_df = df[df['category'] == category]
        # Resample by month ('M') to reduce clutter
        sentiment_trends = category_df['sentiment_textblob'].resample('M').mean()
        plt.plot(sentiment_trends.index, sentiment_trends, label=category, marker='o')  # Adding markers for readability
    
    plt.title('Sentiment Trends Over Time (Monthly)', color='orange')
    plt.xlabel('Date', color='orange')
    plt.ylabel('Average Sentiment', color='orange')
    plt.legend(title='Product Categories', title_fontsize='13', loc='best')
    plt.gca().set_facecolor('#f4f4f9')
    plt.gca().spines['bottom'].set_color('orange')
    plt.gca().spines['top'].set_color('orange')
    plt.gca().spines['right'].set_color('orange')
    plt.gca().spines['left'].set_color('orange')
    plt.grid(True, color='orange')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#f4f4f9')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def plot_sentiment_heatmap(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'sentiment_textblob'])
    df.set_index('date', inplace=True)
    
    # Resample by month ('M') to reduce clutter
    heatmap_data = df.pivot_table(index=df.index.to_period('M').to_timestamp(), columns='category', values='sentiment_textblob', aggfunc='mean')
    
    plt.figure(figsize=(14, 8), facecolor='#f4f4f9')
    sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt='.2f', cbar=True)
    plt.title('Sentiment Heatmap (Monthly)', color='orange')
    plt.xlabel('Categories', color='orange')
    plt.ylabel('Date', color='orange')
    plt.gca().set_facecolor('#f4f4f9')
    plt.gca().spines['bottom'].set_color('orange')
    plt.gca().spines['top'].set_color('orange')
    plt.gca().spines['right'].set_color('orange')
    plt.gca().spines['left'].set_color('orange')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#f4f4f9')
    img.seek(0)
    heatmap_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return heatmap_url


def plot_network_graph(df, top_n=20):
    G = nx.Graph()

    # Preprocess and count keywords
    all_keywords = []
    for _, row in df.iterrows():
        keywords = preprocess_review(row['review_text']).split()
        all_keywords.extend(keywords)

    keyword_counts = Counter(all_keywords)
    most_common_keywords = [keyword for keyword, _ in keyword_counts.most_common(top_n)]

    # Build graph with filtered keywords
    for _, row in df.iterrows():
        category = row['category']
        G.add_node(category, node_type='category')
        keywords = preprocess_review(row['review_text']).split()
        for keyword in keywords:
            if keyword in most_common_keywords:
                G.add_node(keyword, node_type='keyword')
                G.add_edge(category, keyword)

    pos = nx.spring_layout(G, k=1.0, seed=42)  # Increase k for more spacing
    plt.figure(figsize=(16, 12))  # Larger figure size

    # Draw nodes with different colors based on type
    node_colors = ['#ff6600' if G.nodes[node]['node_type'] == 'category' else '#66b3ff' for node in G.nodes()]
    node_sizes = [700 if G.nodes[node]['node_type'] == 'category' else 350 for node in G.nodes()]  # Larger size for categories

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)  # Thinner and more transparent edges

    # Draw labels with adjusted positions and angles
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', font_weight='bold', verticalalignment='center')

    # Adjust label positions
    for node, (x, y) in pos.items():
        offset = 0.1 if G.nodes[node]['node_type'] == 'category' else -0.1
        pos[node] = (x, y + offset)

    plt.title('Network Graph of Product Categories and Keywords', color='#ff6600')
    plt.gca().set_facecolor('#f0f0f0')  # Light grey background
    plt.gca().spines['bottom'].set_color('#ff6600')
    plt.gca().spines['top'].set_color('#ff6600')
    plt.gca().spines['right'].set_color('#ff6600')
    plt.gca().spines['left'].set_color('#ff6600')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    network_graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return network_graph_url





def plot_comparative_analysis(df):
    df['sentiment_textblob'] = df['sentiment_textblob'].astype(int)
    categories = df['category'].unique()
    category_sentiments = df.groupby('category')['sentiment_textblob'].mean()

    plt.figure(figsize=(12, 8), facecolor='#f4f4f9')
    plt.bar(categories, category_sentiments, color='orange')
    plt.xlabel('Category')
    plt.ylabel('Average Sentiment')
    plt.title('Comparative Sentiment Analysis', color='orange')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#f4f4f9')
    img.seek(0)
    comparative_analysis_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return comparative_analysis_url


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['review_file']
        df = pd.read_csv(file)

        # Preprocess reviews
        df['cleaned_review'] = df['review_text'].apply(preprocess_review)

        # Determine sentiment using TextBlob
        df['sentiment_textblob'] = df['cleaned_review'].apply(determine_sentiment)

        # Generate plots
        sentiment_dist_pie_url = plot_pie_chart(df)
        sentiment_dist_bar_url = plot_bar_chart(df)
        sentiment_trends_url = plot_sentiment_trends(df)
        sentiment_heatmap_url = plot_sentiment_heatmap(df)
        network_graph_url = plot_network_graph(df)
        comparative_analysis_url = plot_comparative_analysis(df)

        return render_template('index.html', sentiment_dist_pie_url=sentiment_dist_pie_url, sentiment_dist_bar_url=sentiment_dist_bar_url, sentiment_trends_url=sentiment_trends_url, sentiment_heatmap_url=sentiment_heatmap_url, network_graph_url=network_graph_url, comparative_analysis_url=comparative_analysis_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
