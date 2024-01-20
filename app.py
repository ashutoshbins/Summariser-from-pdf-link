from flask import Flask, render_template, request
from transformers import pipeline
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import textwrap

app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to extract text from a PDF URL
def extract_text_from_pdf_url(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into lines with line breaks after every n words
def split_text_with_line_break(text, words_per_line=10):
    words = text.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_url = request.form['pdf_url']

        if not pdf_url.startswith('/supremecourt'):
            error_message = "Invalid URL. The URL should start with '/supremecourt'."
            return render_template('index.html', error_message=error_message)

        full_url = 'https://main.sci.gov.in' + pdf_url

        # Extract the text
        text = extract_text_from_pdf_url(full_url)

        # Split the text into chunks of approximately 1024 tokens
        chunks = textwrap.wrap(text, width=1024)

        # Summarize each chunk and combine the summaries
        summary = []
        for chunk in chunks:
            if len(chunk.split()) > 50:
                summary.append(summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text'])
            else:
                summary.append(chunk)

        # Prepare the summarized content for rendering
        summarized_content = []
        for sentence in summary:
            lines = split_text_with_line_break(sentence)
            summarized_content.extend(lines.split('\n'))

        return render_template('index.html', pdf_url=pdf_url, summarized_content=summarized_content)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
