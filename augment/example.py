import random

from augment import base
from data import model


class ExampleAugmentationStep(base.AugmentationStep):
    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        for mention in doc.mentions:
            for token_idx in mention.token_indices:
                if random.random() < .1:
                    doc.tokens[token_idx].text = 'example'

        return doc
