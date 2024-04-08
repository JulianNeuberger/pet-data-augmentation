import copy
import typing
from random import random, randint

from nltk.corpus import wordnet
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import data
from data import model
from transformations import tokenmanager

docs: typing.List[model.Document] = data.loader.read_documents_from_json('../complete.json')
def mask(strr):
    unmasker = pipeline(
        "fill-mask", model="xlm-roberta-base", top_k=1
    )
    masked_input = strr.replace(strr, "<mask>", 1)
    new_str = unmasker("I <mask> to read a <mask>")
    new_words = []
    for nw in new_str:
        new_words.append(nw[0]["token_str"])
    print(new_words)
#mask("make")


def do_augment2(doc2: model.Document) -> model.Document:
    doc = copy.deepcopy(doc2)
    i = 0
    p = 1
    while i < len(doc.sentences):
        if random() > p:
            sent_length = len(doc.sentences[i].tokens) - 1
            if i < len(doc.sentences) - 1:
                # transfer Mentions
                for mention in doc.mentions:
                    if mention.sentence_index == i + 1:
                        mention.sentence_index -= 1
                        for ment_id in mention.token_indices:
                            ment_id += sent_length
                # transfer Tokens
                tok_arr = []
                for j, token in enumerate(doc.sentences[i + 1].tokens):
                    tok = model.Token(token.text, token.index_in_document - 1, token.pos_tag, token.bio_tag,
                                      token.sentence_index - 1)
                    tok_arr.append(tok)
                # delete sentence
                tokenmanager.delete_sentence(doc, i + 1)

                for j, tok in enumerate(tok_arr):
                    tokenmanager.create_token(doc, tok, sent_length + j)

                # delete punct
                tokenmanager.delete_token(doc, tok_arr[-1].index_in_document + 1)
        i += 1
    return doc






def do_augment(doc2: model.Document, count_insertions=10) -> model.Document:
    doc = copy.deepcopy(doc2)
    for sentence in doc.sentences:
        counter = 0
        while counter < count_insertions:
            ran = randint(0, len(sentence.tokens) - 1)
            text = "Test"
            if sentence.tokens[ran].bio_tag == "O":
                bio = sentence.tokens[ran].bio_tag
            else:
                bio = "I-" + tokenmanager.get_bio_tag_short(sentence.tokens[ran].bio_tag)
            tok = model.Token(text, sentence.tokens[ran].index_in_document + 1,
                              tokenmanager.get_pos_tag([text]),
                              bio,
                              sentence.tokens[ran].sentence_index)

            mentions = tokenmanager.get_mentions(doc, ran, sentence.tokens[ran].sentence_index)
            if mentions != []:
                tokenmanager.create_token(doc, tok, ran + 1, mentions[0])
            else:
                tokenmanager.create_token(doc, tok, ran + 1)
            counter += 1
    return doc

#tokenmanager.delete_sentence(docs[0], 0)
print(wordnet.synsets("word"))
#print(do_augment(docs[0]))