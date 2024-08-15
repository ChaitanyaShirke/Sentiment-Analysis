# Sentiment Analysis Portal

## Project Overview

The Sentiment Analysis Portal is a web-based application designed to analyze and visualize sentiment from customer reviews. The portal allows users to upload CSV files containing reviews and other related information and provides various visualizations to better understand sentiment trends, distribution, and keyword associations.

## Project Structure

![image](https://github.com/user-attachments/assets/68d98aeb-4d5b-4b11-aa05-38fd2d9927cd)

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

   Use the following command to install the required dependencies:


   pip install -r requirements.txt
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
