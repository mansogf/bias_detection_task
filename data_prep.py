import re
import nltk
from nltk.stem.porter import PorterStemmer

def data_cleaning(text):
    # Remove symbols and punctuations & apply lower() to string
    formatted_text = re.sub(r"[^\w\s]", " ", text).lower()

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [i for i in formatted_text.split() if not i in stopwords]

    # Stemming tokens
    word_stem = [PorterStemmer().stem(word) for word in words]

    return " ".join(word_stem)
    