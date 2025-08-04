from collections import Counter

from torchtext.data import Dataset


def build_vocab(train_data: Dataset, dev_data: Dataset, test_data: Dataset):

    vocab = Counter()
    for text in train_data.examples:
        vocab.update(text.src)
    for text in dev_data.examples:
        vocab.update(text.src)
    for text in test_data.examples:
        vocab.update(text.src)

    start_symbol = "<start>"
    end_symbol = "<end>"
    pad_symbol = "<pad>"
    vocab.update([start_symbol, end_symbol, pad_symbol])


    word2idx = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}

    return word2idx

