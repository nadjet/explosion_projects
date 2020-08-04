import stanza
from spacy_stanza import StanzaLanguage
import sys
from utils.log import logger
import numpy as np


vectors_loc = sys.argv[1]
vectors_name = 'fasttext'
max_items = -1

config = {
  "lang":"sv",
    "tokenize_model_path": "/Users/nadjet/stanza_resources/sv/tokenize/talbanken.pt",
    "pos_model_path": "/Users/nadjet/stanza_resources/sv/pos/talbanken.pt",
    "pos_pretrain_path": "/Users/nadjet/stanza_resources/sv/pretrain/talbanken.pt",
    "lemma_model_path": "/Users/nadjet/stanza_resources/sv/lemma/talbanken.pt",
    "depparse_model_path": "/Users/nadjet/stanza_resources/sv/depparse/talbanken.pt",
    "depparse_pretrain_path": "/Users/nadjet/stanza_resources/sv/pretrain/talbanken.pt"
}
snlp = stanza.Pipeline(**config)
nlp = StanzaLanguage(snlp)
with open(vectors_loc, 'rb') as file_:
    logger.info("Reading file '{}'".format(vectors_loc))
    header = file_.readline()
    nr_row, nr_dim = header.split()  # the first line is number of tokens and dimensions
    counter = 0
    nlp.vocab.reset_vectors(width=int(nr_dim))
    for line in file_:
        if counter % 100 == 0:
            logger.info(counter)
        if counter == max_items:
            break
        counter = counter + 1
        line = line.rstrip().decode('utf8')
        pieces = line.rsplit(' ', int(nr_dim))
        word = pieces[0]
        vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
        nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
    nlp.vocab.vectors.name = vectors_name  # give vectors a name