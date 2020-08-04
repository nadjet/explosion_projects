import plac
from pathlib import Path
import stanza
from spacy_stanza import StanzaLanguage

import numpy as np


from spacy.lang.sv import Swedish
import spacy

from utils.log import logger


# https://github.com/souvikg10/spacy-fasttext/blob/master/load_fastText.py
@plac.annotations(
    vectors_loc=("Path to the vectors file", "option", "vec", Path),
    lang=("Language. Defaults to 'sv'", "option", "la", str),
    stz=("Whether to use spacy-stanza. Defaults to 'True'", "option", "stz", str),
    vectors_name=("Vectors name. Defaults to 'fasttext'","option","vname",str)
)
def main(vectors_loc=None, lang=None, stz=True, vectors_name='fasttext'):
    return create_model(vectors_loc=vectors_loc, lang=lang, stz=stz, vectors_name=vectors_name)


def create_model(vectors_loc=None, lang=None, stz=True, vectors_name='fasttext', max_items=-1):
    if lang is None or lang=='sv' and not stz:
        nlp = Swedish()
    elif not stz:
        nlp = spacy.blank(lang)
    elif stz:
        stanza.download(lang)
        snlp = stanza.Pipeline(lang=lang)
        nlp = StanzaLanguage(snlp)

    with open(vectors_loc, 'rb') as file_:
        logger.info("Reading file '{}'".format(vectors_loc))
        header = file_.readline()
        nr_row, nr_dim = header.split() # the first line is number of tokens and dimensions
        counter = 0
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in file_:
            if counter%100==0:
                logger.info(counter)
            if counter==max_items:
                break
            counter = counter + 1
            line = line.rstrip().decode('utf8')
            pieces = line.rsplit(' ', int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
            nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
        nlp.vocab.vectors.name = vectors_name # give vectors a name
    return nlp


def analyse_text(text, nlp):
    doc = nlp(text)
    width = 15
    print(f"{'Token': <{width}} {'Lemma': <{width}} {'POS': <{width}}")
    print(f"{'':-<{width}} {'':-<{width}} {'':-<{width}}")
    for token in doc:
        print(f"{token.text: <{width}} {token.lemma_: <{width}} {token.pos_: <{width}}")


if __name__ == "__main__":
    nlp = plac.call(main)
    text = "\"Zlatan Ibrahimović, född 3 oktober 1981 i Västra Skrävlinge församling i Malmö, är en svensk fotbollsspelare. Ibrahimović har tilldelats Guldbollen elva gånger och anses allmänt vara Sveriges bäste fotbollsspelare genom tiderna. Från 2001 till 2016 spelade han i svenska landslaget, där han med sina 62 mål är den främste målgöraren genom tiderna. Under sina 18 år som utlandsproffs har svensken vunnit ligan vid elva tillfällen med fem olika klubbar i fyra länder och blivit skyttekung i italienska Serie A två gånger och i franska Ligue 1 vid tre tillfällen. Ibrahimović är den ende som spelat för sju olika klubbar i Champions League, där han med sina 48 mål också intar en delad niondeplats i skytteligans maratontabell. Hans främsta internationella merit är segern i Europa League med Manchester United 2016/2017. Svenskens övergång från Inter till Barcelona 2009 var den spanska storklubbens dittills dyraste spelarköp (69 miljoner euro). 2015 var Ibrahimović enligt tidskriften Forbes den 55:e bäst betalda kändisen i världen, med en årsinkomst på 39 miljoner dollar. I september 2018 gjorde Ibrahimović sitt 500:e mål och blev därmed en av 28 spelare i fotbollshistorien som gjort minst 500 mål (landslag och klubblag).\""
    analyse_text(text, nlp)
