import typing

from nltk.corpus import wordnet

from augment import base, params
from data import model


class Trafo3Step(base.BaseTokenReplacementStep):
    def get_replacement_candidates(
        self, document: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return [[t] for t in document.tokens if t.pos_tag in ["JJ", "JJR", "JJS"]]

    def get_replacement(
        self, candidate: typing.List[model.Token]
    ) -> typing.Optional[typing.List[str]]:
        text = " ".join(t.text for t in candidate)

        syn_sets = wordnet.synsets(text, "a")
        syn_sets = [s for s in syn_sets if ".a." in s.name()]

        if len(syn_sets) == 0:
            return None

        first_syn_set = syn_sets[0]
        lemma = first_syn_set.lemmas()[0]
        antonyms = lemma.antonyms()

        if len(antonyms) == 0:
            return None

        antonyms.sort(key=lambda x: str(x).split(".")[2])
        antonym = antonyms[0].name()
        return antonym.split("_")

    def get_params(self) -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(
                name="replacements_per_sentence", min_value=1, max_value=20
            ),
        ]
