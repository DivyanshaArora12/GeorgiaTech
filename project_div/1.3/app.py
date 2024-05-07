from flask import Flask, render_template, request # type: ignore
import os
from sec_edgar_downloader import Downloader # type: ignore
import re
import matplotlib.pyplot as plt # type: ignore
from transformers import T5ForConditionalGeneration, T5Tokenizer # type: ignore

app = Flask(__name__)

# Initialize the SEC Edgar Downloader
dl = Downloader("My Company", "example@email.com", "sec-edgar-filings")

# Function to extract revenue data from 10-K filing text
def extract_revenue_data(filing_text):
    # Regular expression pattern to match revenue figures
    revenue_pattern = r"(?i)\b(revenue|sales|income)\b.*?\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b"
    
    # Find all matches of revenue pattern in the filing text
    revenue_matches = re.findall(revenue_pattern, filing_text)
    
    # Extract revenue figures from matches
    revenues = []
    for match in revenue_matches:
        revenue = float(match[1].replace(",", ""))  # Convert revenue string to float
        revenues.append(revenue)
    
    return revenues

# Function to generate insights from 10-K filings using T5 model
def generate_insight(filing_text):
    # Load pre-trained T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Preprocess the filing text
    prompt = "Generate insight about revenue growth based on the provided 10-K filing text: "
    input_text = prompt + filing_text[:2048]  # Limit input text to avoid exceeding maximum token length

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate insight using the T5 model
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated insight
    generated_insight = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_insight

# Function to visualize revenue data for a specific year
def visualize_revenue_growth(year, revenues):
    # Plot revenue data for the given year
    plt.figure(figsize=(8, 6))
    plt.plot(revenues, marker='o', color='b', linestyle='-')
    plt.title(f'Company XYZ Revenue Growth ({year})')
    plt.xlabel('Quarter')
    plt.ylabel('Revenue (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"static/revenue_growth_{year}.png")  # Save the plot as a PNG file
    plt.close()

# Function to recursively search for a text file in a directory
def find_text_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                return os.path.join(root, file)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    ticker = request.form['ticker']
    year = request.form['year']

    # Download the 10-K filing for the specified year
    dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{int(year)+1}-01-01")

    # Search for the text file in the specified directory
    root_dir = os.path.join("sec-edgar-filings","sec-edgar-filings", ticker, "10-K")
    filing_path = find_text_file(root_dir)

    # Read the downloaded filing text
    with open(filing_path, "r", encoding="utf-8") as file:
        filing_text = file.read()

    # Generate insight from the filing text
    insight = generate_insight(filing_text)

    # Extract revenue data from the filing text
    revenues = extract_revenue_data(filing_text)

    # Visualize revenue growth for the specific year
    visualize_revenue_growth(year, revenues)

    return render_template('results.html', insight=insight, ticker=ticker, year=year)

if __name__ == '__main__':
    app.run(debug=True)
