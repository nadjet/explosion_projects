from spacy.pipeline import EntityRuler
from spacy.kb import KnowledgeBase
from utils.vec2model import create_model
from spacy.vocab import Vocab

class NamedEntityCreator:

    def __init__(self, kb_folder, vectors_loc, lang='sv', stz=True, vectors_name='fasttext'):
        self.nlp = create_model(vectors_loc=vectors_loc, lang=lang, stz=stz, vectors_name=vectors_name, max_items=1000)
        self.kb = KnowledgeBase(vocab=self.nlp.vocab)
        print(kb_folder)
        self.kb.load_bulk(kb_folder)
        print()
        _print_kb(self.kb)


def _print_kb(kb):
    print(kb.get_size_entities(), "kb entities:", kb.get_entity_strings())
    print(kb.get_size_aliases(), "kb aliases:", kb.get_alias_strings())

import sys
nec = NamedEntityCreator(sys.argv[1],sys.argv[2])
all_entities = nec.kb.get_entity_strings()
print(type(all_entities))
print (all_entities)