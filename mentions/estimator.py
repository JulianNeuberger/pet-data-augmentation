import os
import pathlib
import typing

import pycrfsuite

from data import PetDocument, PetToken, PetMention
from eval import metrics
from mentions import decoder


class ConditionalRandomFieldsEstimator:
    def __init__(self, model_file_path: pathlib.Path):
        self._model_path = model_file_path

    @staticmethod
    def load(path: str):
        tagger = pycrfsuite.Tagger()
        tagger.open(path)
        return tagger

    @staticmethod
    def load_model(model_name: str):
        estimator = ConditionalRandomFieldsEstimator(
            pathlib.Path(f"api/models/crf/{model_name}")
        )
        estimator.tagger = pycrfsuite.Tagger()
        estimator.tagger.open(str(estimator._model_path))
        return estimator

    def train(self, train_documents: typing.List[PetDocument]) -> pycrfsuite.Tagger:
        trainer = pycrfsuite.Trainer(verbose=False)

        for train_document in train_documents:
            X_train = [
                self._features_from_tokens(ts) for ts in train_document.sentences
            ]
            y_train = self._labels_from_tokens(
                train_document.tokens, train_document.mentions
            )
            assert len(X_train) == len(y_train)
            assert all([len(xs) == len(ys) for xs, ys in zip(X_train, y_train)])

            for xseq, yseq in zip(X_train, y_train):
                trainer.append(xseq, yseq)

        trainer.set_params(
            {
                "c1": 1.0,  # coefficient for L1 penalty
                "c2": 1e-3,  # coefficient for L2 penalty
                "max_iterations": 100,  # 50,  # stop earlier
                # include transitions that are possible, but not observed
                "feature.possible_transitions": True,
            }
        )

        os.makedirs(str(self._model_path.parent), exist_ok=True)
        trainer.train(str(self._model_path))

        return self.load(str(self._model_path))

    def predict(
        self, test_documents: typing.List[PetDocument]
    ) -> typing.List[PetDocument]:
        predicted_documents = []
        for test_document in test_documents:
            tagger = self.load(str(self._model_path))
            X_test = [self._features_from_tokens(s) for s in test_document.sentences]
            y_pred = [tagger.tag(xseq) for xseq in X_test]
            predicted = decoder.decode_predictions(test_document, y_pred)
            predicted_documents.append(predicted)

        return predicted_documents

    def test(
        self, test_documents: typing.List[PetDocument]
    ) -> typing.Dict[str, metrics.Stats]:
        predicted_documents = self.predict(test_documents)
        ground_truth_documents = []

        for test_document in test_documents:
            y_test = self._labels_from_tokens(
                test_document.tokens, test_document.mentions
            )
            ground_truth = decoder.decode_predictions(test_document, y_test)
            ground_truth_documents.append(ground_truth)

        return metrics.mentions_f1_stats(
            predicted_documents=predicted_documents,
            ground_truth_documents=ground_truth_documents,
        )

    @staticmethod
    def _features_from_tokens(
        tokens: typing.List[PetToken],
    ) -> typing.List[typing.List[str]]:
        return [
            ConditionalRandomFieldsEstimator._features_from_token(tokens, i)
            for i, _ in enumerate(tokens)
        ]

    @staticmethod
    def _features_from_token(
        tokens: typing.List[PetToken], token_index: int
    ) -> typing.List[str]:
        word = tokens[token_index].text
        pos_tag = tokens[token_index].pos_tag

        features = [
            "bias",
            "word.lower=" + word.lower(),
            "word[-3:]=" + word[-3:],
            "word[-2:]=" + word[-2:],
            "word.isupper=%s" % word.isupper(),
            "word.istitle=%s" % word.istitle(),
            "word.isdigit=%s" % word.isdigit(),
            "postag=" + pos_tag,
            "postag[:2]=" + pos_tag[:2],
        ]
        if token_index > 0:
            word1 = tokens[token_index - 1].text
            postag1 = tokens[token_index - 1].pos_tag
            features.extend(
                [
                    "-1:word.lower=" + word1.lower(),
                    "-1:word.istitle=%s" % word1.istitle(),
                    "-1:word.isupper=%s" % word1.isupper(),
                    "-1:postag=" + postag1,
                    "-1:postag[:2]=" + postag1[:2],
                ]
            )
        else:
            features.append("BOS")

        if token_index < len(tokens) - 1:
            word1 = tokens[token_index + 1].text
            postag1 = tokens[token_index + 1].pos_tag
            features.extend(
                [
                    "+1:word.lower=" + word1.lower(),
                    "+1:word.istitle=%s" % word1.istitle(),
                    "+1:word.isupper=%s" % word1.isupper(),
                    "+1:postag=" + postag1,
                    "+1:postag[:2]=" + postag1[:2],
                ]
            )
        else:
            features.append("EOS")

        return features

    @staticmethod
    def _labels_from_tokens(
        tokens: typing.List[PetToken], mentions: typing.List[PetMention]
    ) -> typing.List[typing.List[str]]:
        labels: typing.List[typing.List[str]] = []
        sentence_id: typing.Optional[int] = None
        last_mention: typing.Optional[PetMention] = None
        for token in tokens:
            tag: typing.Optional[str] = None
            first_in_mention = True
            for mention in mentions:
                if token.index_in_document in mention.token_document_indices:
                    tag = mention.type
                    if last_mention == mention:
                        first_in_mention = False
                    last_mention = mention
                    break
            if token.sentence_index != sentence_id:
                sentence_id = token.sentence_index
                labels.append([])
            if tag is None:
                labels[-1].append("O")
            else:
                if first_in_mention:
                    labels[-1].append(f"B-{tag}")
                else:
                    labels[-1].append(f"I-{tag}")
        return labels

    @staticmethod
    def _print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    @staticmethod
    def _print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))
