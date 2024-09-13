import json
import re
import typing

import nltk.tokenize
from openai import OpenAI

from augment import base
from data import PetToken, PetDocument


class LargeLanguageModelRephrasing(base.BaseTokenReplacementStep):
    # TODO: load llm host etc from env...
    def __init__(
        self,
        dataset: typing.List[PetDocument],
        replace_probability: float,
        llm_host: str = "http://132.180.195.1:8007/v1",
        api_key: str = "ollama",
        model: str = "llama3.1:70b",
        **kwargs,
    ):
        super().__init__(dataset, replace_probability, **kwargs)

        self.client = OpenAI(
            base_url=llm_host,
            api_key=api_key,  # required, but unused
        )
        self.model = model

    def get_replacement_candidates(
        self, doc: PetDocument
    ) -> typing.List[typing.List[PetToken]]:
        return self.get_sequences(doc)

    def get_replacements(
        self,
        candidate: typing.List[PetToken],
        num_replacements_per_candidate: int,
        document: PetDocument,
    ) -> typing.List[typing.List[str]]:
        texts = [t.text for t in document.tokens]
        texts.insert(candidate[-1].index_in_document + 1, "##")
        texts.insert(candidate[0].index_in_document + 1, "##")

        candidate_text = " ".join(t.text for t in candidate)
        document_text = " ".join(t.text for t in document.tokens)

        prompt = (
            f"Give a list of {num_replacements_per_candidate} possible alternatives "
            f'for "{candidate_text}" in the following text. '
            f"Format your response as json list and only return the list without any "
            f'additional text.\n "{document_text}"'
        )

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
        )
        raw_text = response.choices[0].text

        # remove any code formatting that may occur in some models...
        if "```" in raw_text:
            raw_text = re.sub(r"```\w*", "", raw_text)

        replacements = []
        try:
            rephrased_texts = json.loads(raw_text)

            if isinstance(rephrased_texts, list):
                if isinstance(rephrased_texts[0], dict):
                    if len(rephrased_texts[0]) == 1:
                        rephrased_texts = [list(o.values())[0] for o in rephrased_texts]

            tokenized_texts = [nltk.tokenize.word_tokenize(t) for t in rephrased_texts]
            if any(len(t) == 0 for t in tokenized_texts):
                print(f"Zero length from: {rephrased_texts}")
            tokenized_texts = [t for t in tokenized_texts if len(t) > 0]
            replacements = tokenized_texts

        except json.JSONDecodeError:
            print(
                f'LLM "{self.model}" did not follow format instructions: "{raw_text}" is not valid json.'
            )
        except TypeError:
            print(f'Response contains invalid objects: "{rephrased_texts}"')

        return replacements
