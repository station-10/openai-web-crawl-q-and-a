################################################################################
# Step 1
################################################################################

import requests
import time
import re
import ast
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# set openai api key
with open('openai_api.key', 'r') as file:
    my_key = file.read().replace('\n', '')
openai.api_key = my_key

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Define root domain to crawl
domain = "experienceleague.adobe.com"
# Define path prefix to limit the crawl so we don't get the entirely of Adobe's documentation
path_prefix = "/docs/experience-platform/edge/"
# Define a URL to start the crawl
url = "https://experienceleague.adobe.com/docs/experience-platform/edge/home.html"

# Set the desired delay and chunk size for get_embeddings function
# This should work with the free tier of the OpenAI API, you could increase this if you have a paid account
# https://platform.openai.com/docs/guides/rate-limits/overview
delay = 60
text_chunks = 20


# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
# Step 2
################################################################################

# Function to get the hyperlinks from a URL


def get_hyperlinks(url):

    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
# Step 3
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
# Updated this script to exclude # links and links that don't start with the path_prefix


def get_domain_hyperlinks(domain, path_prefix, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain and has the desired path prefix
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == domain and url_obj.path.startswith(path_prefix):
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                clean_link = "https://" + domain + link
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            else:
                clean_link = "https://" + domain + path_prefix + link

            # Check if the resulting link still has the desired path prefix
            url_obj = urlparse(clean_link)
            if not url_obj.path.startswith(path_prefix):
                clean_link = None

        # Check if the clean_link contains a '#' and if so, set it to None
        if clean_link is not None and "#" in clean_link:
            clean_link = None

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain and have the desired path prefix
    return list(set(clean_links))

################################################################################
# Step 4
################################################################################

# Updated this script to only scrape the body of the Adobe documentation pages
# Otherwise the training data is full of header and footer text, which we don't need
# Note this is specific to Adobe documentation pages


def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # Find the div with data-id="body"
            body_div = soup.find('div', {'data-id': 'body'})

            if body_div:
                # Get the text but remove the tags
                text = body_div.get_text()

                # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                if "You need to enable JavaScript to run this app." in text:
                    print("Unable to parse page " + url +
                          " due to JavaScript being required")

                # Otherwise, write the text to the file in the text directory
                f.write(text)

            else:
                print("Unable to find div with data-id='body' in URL: " + url)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(domain, path_prefix, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)


crawl(url)


################################################################################
# Step 5
################################################################################


def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


################################################################################
# Step 6
################################################################################

# Create a list to store the text files
texts = []

# Get all the text files in the text directory
for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append(
            (file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns=['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

################################################################################
# Step 7
################################################################################

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

################################################################################
# Step 8
################################################################################

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens


def split_into_many(text, max_tokens=max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence))
                for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])

################################################################################
# Step 9
################################################################################

df = pd.DataFrame(shortened, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

################################################################################
# Step 10
################################################################################

# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits


def get_embedding(text):
    response = openai.Embedding.create(
        input=text, engine='text-embedding-ada-002')
    return response['data'][0]['embedding']


embeddings = []
counter = 0

for text in df.text:
    try:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    except Exception as e:
        if 'RateLimitError' in str(e):
            print(
                "Rate limit exceeded. Waiting {} seconds before retrying...".format(delay))
            time.sleep(delay)
            embedding = get_embedding(text)
            embeddings.append(embedding)
        else:
            print("Error: ", e)
            embeddings.append(None)
    counter += 1
    if counter % text_chunks == 0:
        print("Processed {} texts. Waiting {} seconds before continuing...".format(
            counter, delay))
        time.sleep(delay)

df['embeddings'] = embeddings
df.to_csv('processed/embeddings.csv')
df.head()

################################################################################
# Step 11
################################################################################


def convert_to_array(embedding_str):
    if pd.isna(embedding_str):
        return None
    return np.array(ast.literal_eval(embedding_str))


df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(convert_to_array)

df.head()

################################################################################
# Step 12
################################################################################


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    q_embeddings = np.array(q_embeddings).reshape(1, -1)

    # Get the distances from the embeddings
    def calculate_distance(x):
        if x is None:
            return float('inf')
        return cosine_distances(q_embeddings, x.reshape(1, -1)).item()

    df['distances'] = df['embeddings'].apply(calculate_distance)

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="text-davinci-003",
    question="placeholder question",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

################################################################################
# Step 13
################################################################################


print(answer_question(
    df,
    question="What does the 'edgeConfigId' part of the alloy 'configure' command represent?",
    debug=False))

################################################################################
# Step 14 - WORK IN PROGRESS
################################################################################
# Experimenting with the GPT-3.5 version of the api
# see discussion here around how to use embeddings and context with the new version of the api
# https://community.openai.com/t/how-can-i-use-embeddings-with-chat-gpt-3-5-turbo/86759/24


def answer_question_gpt_3_5(
    question="placeholder question",
    max_len=1800,
    size="ada",
    debug=False,
):
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"},
                {"role": "user", "content": question},
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""


# call the GPT-3.5 function
print(answer_question_gpt_3_5(
    df,
    question="What does the 'edgeConfigId' part of the alloy 'configure' command represent?",
    debug=False))
