# Sentiment Analysis Portal

## Project Overview

The Sentiment Analysis Portal is a web-based application designed to analyze and visualize sentiment from customer reviews. The portal allows users to upload CSV files containing reviews and other related information and provides various visualizations to better understand sentiment trends, distribution, and keyword associations.

## Project Structure
![image](https://github.com/user-attachments/assets/dc8de583-cbee-4133-88c5-3c83683db8ec)
### Explanation of Files and Directories:

- **static/css/styles.css**: Stylesheets for the frontend.
- **static/js/output.js**: JavaScript code handling interactivity and form validation.
- **templates/index.html**: The main HTML template for the user interface.
- **models/sentiment_model.pkl**: The trained sentiment analysis model.
- **models/vectorizer.pkl**: The vectorizer used for text data transformation.
- **app.py**: The Flask backend application managing file uploads, sentiment analysis, and visualization generation.

## Installation and Setup

### Prerequisites

- Python 3.x
- Flask
- Required Python packages (refer to `requirements.txt`)

### Steps to Set Up Locally

1. **Clone the Repository**:

   git clone https://github.com/ChaitanyaShirke/SentimentAnaalysisSle.git
   cd SentimentAnalysisSle

2. **Install Dependencies**:

   - **Option 1: Using `requirements.txt`**:
   
     Install the required dependencies using the command:
     
     ```bash
     pip install -r requirements.txt
     ```

   - **Option 2: Manually Install Required Libraries**:
   
     If `requirements.txt` fails or is unavailable, you can manually install the necessary libraries with:
     
     ```bash
      pip install Flask>=2.1.1,<3.0 numpy>=1.24.2,<2.0 pandas>=1.5.3,<2.0 scikit-learn>=1.1.1,<2.0 matplotlib>=3.7.1,<4.0 plotly>=5.9.0,<6.0 d3js==7.8.4 seaborn>=0.11,<0.12 textblob>=0.15,<0.16 nltk>=3.6,<4.0 networkx>=2.5,<3.0 wordcloud>=1.8,<2.0
     ```

3. **Run the Application**:

   Start the Flask application by running:

   python app.py

4. **Access the Portal**:

   Open your web browser and go to `http://127.0.0.1:5000/` to use the Sentiment Analysis Portal.


---

### Download and Install

You can download the project directly from the GitHub repository:

[Download Sentiment Analysis Portal](https://github.com/ChaitanyaShirke/SentimentAnaalysisSle/archive/refs/heads/main.zip)
