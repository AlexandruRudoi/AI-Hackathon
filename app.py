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
        
        # API endpoint for content extraction
        url_cont_extr = "http://localhost:8989/rest/process"
        header = {"accept": "application/json", "Content-Type": "application/json"}
        payload = {"content": url,"language":"EMPTY"}
        response = requests.post(url_cont_extr, headers=header, json=payload)
        # Extract the news text from the URL
        if response.status_code == 200:
            text = response.json()['text']
            print(text)
        else:
            print("Error:", response.text)  # Print the error message if the request failed




        return redirect(url_for('home'))
    else: 
        return render_template("index.html", images=images)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# Run the R script asynchronously(executes the script in the background)
# subprocess.Popen(["Rscript", "your_script.R"])
# Run the R script synchronously(wait for the script to finish)
# subprocess.run(["Rscript", "your_script.R"])