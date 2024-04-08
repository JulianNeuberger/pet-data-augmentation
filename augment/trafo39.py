import random
import typing

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo39Step(base.AugmentationStep):
    """
    Based on https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/entity_mention_replacement_ner/transformation.py
    Only works per sentence, as there is no inter-document information
    about entities (opposed to, e.g., a person mentioned multiple times
    in different documents).
    """

    def __init__(self, dataset: typing.List[model.Document], n: int = 10):
        super().__init__(dataset)
        self.n = n

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.IntegerParam(name="n", min_value=1, max_value=20)]

    @staticmethod
    def extract_entity_mentions(
        doc: model.Document,
    ) -> typing.Dict[int, typing.List[model.Mention]]:
        mentions = {}

        for entity_id, entity in enumerate(doc.entities):
            mentions[entity_id] = []

            for mention_id in entity.mention_indices:
                mentions[entity_id].append(doc.mentions[mention_id])

        return mentions

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        candidates = self.extract_entity_mentions(doc)
        candidates = {k: v for k, v in candidates.items() if len(v) > 1}

        num_changes = 0

        for entity_id, mentions in candidates.items():
            if len(mentions) < 2:
                continue

            target_mention = mentions.pop(random.randrange(0, len(mentions)))
            source_mention: model.Mention = random.choice(mentions)

            target_mention_index = doc.mention_index(target_mention)
            new_text = [doc.tokens[i].text for i in source_mention.token_indices]

            tokenmanager.replace_mention_text(doc, target_mention_index, new_text)

            num_changes += 1

            if num_changes == self.n:
                break

        return doc
