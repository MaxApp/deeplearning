import re
# import nltk

def clean_and_tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    # data = nltk.word_tokenize(data)
    data = [ch.lower() for ch in data]
    return data


def get_sliding_context(tokenized_words: list[str], half_context_size: int):
    """A sliding window generator"""
    i = half_context_size
    while i < len(tokenized_words) - half_context_size:
        center_word = tokenized_words[i]
        context_words = tokenized_words[(i - half_context_size):i] + tokenized_words[(i+1):(i+half_context_size+1)]
        yield context_words, center_word
        i += 1


# test code
# a = ['which', 'team', 'is', 'the', '``', 'champion', "''", 'of', 'the', 'world', 'cup', '2026', '.', '❤️', 'espana', '.']
# for cont, word in get_sliding_context(a, 3):
#     print(f"context: {cont}  center: {word}")