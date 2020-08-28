import re
from collections import Counter
from lib import Pymorphy2Lemmatizer

def preprocess(text, big=True):

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace('-', '')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('\n', '')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    lemmatizer = Pymorphy2Lemmatizer()
    
    lemmatized = [lemmatizer.transform_token(word) for word in words]
    
    # Remove all words with  5 or fewer occurences
    #word_counts = Counter(words)
    #trimmed_words = [word for word in words if word_counts[word] > 5]
    if big:
        word_counts = Counter(lemmatized)
        trimmed_words = [word for word in lemmatized if word_counts[word] > 5]

        return trimmed_words
    else:
        return lemmatized


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

