# utils/text_utils.py

from nltk.corpus import stopwords
import spacy
import nltk

# Download stopwords from NLTK
nltk.download('stopwords')

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# NLTK stop words
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    """
    # Lowercase
    text = text.lower()
    
    # Remove stop words and lemmatize
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if token.text not in stop_words]
    
    # Join lemmatized tokens back into a string
    clean_text = ' '.join(lemmatized)
    
    return clean_text
