from flask import Flask, render_template, request, redirect, url_for
import subprocess
from bs4 import BeautifulSoup
import requests
import csv

# Create a Flask app
app = Flask(__name__)

# Sample list of image URLs
images = []

# Define a route for GET requests
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        url = request.form['link']

        # Extract the news text from the URL
        header = {"accept": "application/json", "Content-Type": "application/json"}
        payload = {"content": url,"language":"EMPTY"}

        # API endpoint for meta data extraction
        url_meta_extr = "http://localhost:8990/rest/process"
        response_meta = requests.post(url_meta_extr, headers=header, json=payload)
        # Handle the response from the API
        if response_meta.status_code == 200:
            meta = response_meta.json()
            # print(meta)
        else:
            print("Error:", response_meta.text)

        # API endpoint for content extraction
        url_cont_extr = "http://localhost:8991/rest/process"
        response_cont = requests.post(url_cont_extr, headers=header, json=payload)
        # Handle the response from the API
        if response_cont.status_code == 200:
            text = response_cont.json()['text']
            # print(text)
        else:
            print("Error:", response_cont.text)  # Print the error message if the request failed
        
        # Write text to a file
        with open('file.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        

        # API endpoint for entity extraction
        payload1 = {"content": text,"language":"xxx"}
        url_entity = "http://localhost:8992/rest/process"
        response_entity = requests.post(url_entity, headers=header, json=payload1)
        # Handle the response from the API
        if response_entity.status_code == 200:
            entity = response_entity.json()
            # print(entity)
        else:
            print("Error:", response_entity.text)
        
        # API endpoint for sentiment analysis
        payload2 = {"content": text,"language":"EMPTY"}
        url_sentiment = "http://localhost:8993/rest/process"
        response_sentiment = requests.post(url_sentiment, headers=header, json=payload2)
        # Handle the response from the API
        if response_sentiment.status_code == 200:
            sentiment = response_sentiment.json()
            sentiment_score = sentiment['score']
            # print(sentiment)
        else:
            print("Error:", response_sentiment.text)
        
        # Quality check for the extracted data
        url_quality = "http://localhost:8989/rest/process"
        response_quality = requests.post(url_quality, headers=header, json=payload)
        # Handle the response from the API
        if response_quality.status_code == 200:
            quality = response_quality.json()
            # print(quality)
        else:
            print("Error:", response_quality.text)
        
        # Run the R script synchronously(wait for the script to finish)
        # arg = "file.txt"
        # subprocess.run(["Rscript", "sentiment_proto_8_single_point.R", arg], capture_output=True, text=True)
        
        # Extracting variables from the quality response
        quality_score = quality['qualityScore']
        # Extracting variables from the quality response
        try:
            author_name = quality['scoreComponents'][0]['meta']['authorName']
        except KeyError:
            author_name = None

        try:
            author_score = int(quality['scoreComponents'][0]['score'] * 100)
        except (KeyError, ValueError):
            author_score = 0

        try:
            entity_score = int(quality['scoreComponents'][1]['score'] * 100)
        except (KeyError, ValueError):
            entity_score = 0

        try:
            sentiment_label = quality['scoreComponents'][2]['meta']['sentimentLabel']
        except KeyError:
            sentiment_label = None

        try:
            sentiment_score = int(quality['scoreComponents'][2]['score'] * 100)
        except (KeyError, ValueError):
            sentiment_score = 0

        
        # Pass the variables to the template
        return render_template("index.html", quality_score=quality_score, author_name=author_name, author_score=author_score, entity_score=entity_score, sentiment_label=sentiment_label, sentiment_score=sentiment_score)
        # return redirect(url_for('home'))
    else: 
        return render_template("index.html")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
