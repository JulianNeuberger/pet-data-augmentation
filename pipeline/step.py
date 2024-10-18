import abc
import dataclasses
import math
import pathlib
import time
import typing

import coref
import mentions
import relations
from data import PetDocument
from eval import metrics


@dataclasses.dataclass
class PipelineStepResult:
    predictions: typing.List[PetDocument]
    stats: typing.Dict[str, metrics.Stats]


T = typing.TypeVar("T")


class PipelineStep(abc.ABC, typing.Generic[T]):
    def __init__(self, name: str, **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if not type(other) == PipelineStep:
            return False
        return self._name == other._name

    def run(
        self,
        *,
        train_documents: typing.List[PetDocument],
        test_documents: typing.List[PetDocument],
        ground_truth_documents: typing.List[PetDocument],
    ):
        train_data = [d.copy(clear=[]) for d in train_documents]
        test_data = [d.copy(clear=[]) for d in test_documents]
        estimator = self.train(train_data)
        result = self.predict(test_data, estimator)
        print("Running evaluation...")
        start = time.time_ns()
        stats = self.eval(ground_truth=ground_truth_documents, predictions=result)
        print(f"Evaluation done after {(time.time_ns() - start) / 1e6:.1f}ms!")
        return PipelineStepResult(result, stats)

    def eval(
        self,
        *,
        predictions: typing.List[PetDocument],
        ground_truth: typing.List[PetDocument],
    ) -> typing.Dict[str, metrics.Stats]:
        raise NotImplementedError()

    def train(self, train_documents: typing.List[PetDocument]) -> T:
        raise NotImplementedError()

    @staticmethod
    def predict(
        test_documents: typing.List[PetDocument], estimator: T
    ) -> typing.List[PetDocument]:
        raise NotImplementedError()


class CatBoostRelationExtractionStep(PipelineStep):
    def __init__(
        self,
        *,
        name: str,
        num_trees: int = 1000,
        negative_sampling_rate: float = 40,
        context_size: int = 2,
        depth: int = 8,
        num_passes: int = 1,
        learning_rate: float = None,
        use_pos_features: bool = False,
        use_embedding_features: bool = False,
        verbose: bool = False,
        class_weighting: float = 0.0,
        seed: int = 42,
        device: str = None,
        device_ids: str = None,
    ):
        super().__init__(name)
        self._num_trees = num_trees
        self._num_passes = num_passes
        self._negative_sampling = negative_sampling_rate
        self._context_size = context_size
        self._verbose = verbose
        self._seed = seed
        self._depth = depth
        self._use_pos_features = use_pos_features
        self._use_embedding_features = use_embedding_features
        self._learning_rate = learning_rate
        self._class_weighting = class_weighting
        self._device = device
        self._device_ids = device_ids

    def eval(
        self,
        *,
        predictions: typing.List[PetDocument],
        ground_truth: typing.List[PetDocument],
    ) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(
            predicted_documents=predictions,
            ground_truth_documents=ground_truth,
            print_only_tags=None,
            verbose=self._verbose,
        )

    def train(
        self, train_documents: typing.List[PetDocument]
    ) -> relations.CatBoostRelationEstimator:
        ner_tags = [
            "Activity",
            "Actor",
            "Activity Data",
            "Condition Specification",
            "Further Specification",
            "AND Gateway",
            "XOR Gateway",
        ]
        relation_tags = [
            "Flow",
            "Uses",
            "Actor Performer",
            "Actor Recipient",
            "Further Specification",
            "Same Gateway",
        ]
        class_weights = {t.lower(): 0 for t in relation_tags}
        if self._class_weighting != 0.0:
            for d in train_documents:
                for r in d.relations:
                    class_weights[r.type.lower()] += 1
            num_samples = sum(class_weights.values())
            num_classes = len(relation_tags)
            class_weights = {
                k: num_samples / (num_classes * v) for k, v in class_weights.items()
            }
            class_weights = {
                k: math.pow(v, 1 / self._class_weighting)
                for k, v in class_weights.items()
            }
        else:
            class_weights = {k: 1.0 for k, v in class_weights.items()}
        print(f"Using class weights {class_weights}")
        estimator = relations.CatBoostRelationEstimator(
            negative_sampling_rate=self._negative_sampling,
            num_trees=self._num_trees,
            use_pos_features=self._use_pos_features,
            use_embedding_features=self._use_embedding_features,
            num_passes=self._num_passes,
            context_size=self._context_size,
            relation_tags=relation_tags,
            ner_tags=ner_tags,
            name=self._name,
            seed=self._seed,
            depth=self._depth,
            learning_rate=self._learning_rate,
            class_weights=class_weights,
            verbose=True,
            device=self._device,
            device_ids=self._device_ids,
        )
        estimator.train(train_documents)
        return estimator

    @staticmethod
    def predict(
        test_documents: typing.List[PetDocument],
        estimator: relations.CatBoostRelationEstimator,
    ) -> typing.List[PetDocument]:
        test_documents = [d.copy(clear=["relations"]) for d in test_documents]
        return estimator.predict(test_documents)


class RuleBasedRelationExtraction(PipelineStep):
    def eval(
        self,
        *,
        predictions: typing.List[PetDocument],
        ground_truth: typing.List[PetDocument],
    ) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(
            predicted_documents=predictions,
            ground_truth_documents=ground_truth,
            print_only_tags=None,
        )

    def train(
        self, train_documents: typing.List[PetDocument]
    ) -> relations.RuleBasedRelationEstimator:
        activity = "Activity"
        actor = "Actor"
        activity_data = "Activity Data"
        condition = "Condition Specification"
        further_spec = "Further Specification"
        and_gateway = "AND Gateway"
        xor_gateway = "XOR Gateway"

        flow = "Flow"
        uses = "Uses"
        performer = "Actor Performer"
        recipient = "Actor Recipient"
        further_spec_relation = "Further Specification"
        same_gateway = "Same Gateway"

        extractor = relations.RuleBasedRelationEstimator(
            [
                relations.rules.SameGatewayRule(
                    triggering_elements=[xor_gateway, and_gateway],
                    target_tag=same_gateway,
                ),
                relations.rules.GatewayActivityRule(
                    gateway_tags=[and_gateway, xor_gateway],
                    activity_tag=activity,
                    same_gateway_tag=same_gateway,
                    flow_tag=flow,
                ),
                relations.rules.SequenceFlowsRule(
                    triggering_elements=[activity, xor_gateway, and_gateway, condition],
                    target_tag=flow,
                ),
                relations.rules.ActorPerformerRecipientRule(
                    actor_tag=actor,
                    activity_tag=activity,
                    performer_tag=performer,
                    recipient_tag=recipient,
                ),
                relations.rules.FurtherSpecificationRule(
                    further_specification_element_tag=further_spec,
                    further_specification_relation_tag=further_spec_relation,
                    activity_tag=activity,
                ),
                relations.rules.UsesRelationRule(
                    activity_data_tag=activity_data,
                    activity_tag=activity,
                    uses_relation_tag=uses,
                ),
            ]
        )
        return extractor

    @staticmethod
    def predict(
        test_documents: typing.List[PetDocument],
        estimator: relations.RuleBasedRelationEstimator,
    ) -> typing.List[PetDocument]:
        test_documents = [d.copy(clear=["relations"]) for d in test_documents]
        return estimator.predict(test_documents)


class CrfMentionEstimatorStep(PipelineStep):

    def eval(
        self,
        *,
        predictions: typing.List[PetDocument],
        ground_truth: typing.List[PetDocument],
    ) -> typing.Dict[str, metrics.Stats]:
        return metrics.mentions_f1_stats(
            predicted_documents=predictions,
            ground_truth_documents=ground_truth,
            print_only_tags=None,
        )

    def train(
        self, train_documents: typing.List[PetDocument]
    ) -> mentions.ConditionalRandomFieldsEstimator:
        estimator = mentions.ConditionalRandomFieldsEstimator(
            pathlib.Path(f"models/crf/{self._name}")
        )
        estimator.train(train_documents)
        return estimator

    @staticmethod
    def predict(
        documents: typing.List[PetDocument],
        estimator: mentions.ConditionalRandomFieldsEstimator,
    ) -> typing.List[PetDocument]:
        mention_extraction_input = [d.copy(clear=["mentions"]) for d in documents]
        return estimator.predict(mention_extraction_input)


class NeuralCoReferenceResolutionStep(PipelineStep):
    def __init__(
        self,
        name: str,
        resolved_tags: typing.List[str],
        ner_strategy: str,
        mention_overlap: float,
        cluster_overlap: float,
    ):
        super().__init__(name)
        self._resolved_tags = resolved_tags
        self._ner_strategy = ner_strategy
        self._mention_overlap = mention_overlap
        self._cluster_overlap = cluster_overlap

    def eval(
        self,
        *,
        predictions: typing.List[PetDocument],
        ground_truth: typing.List[PetDocument],
    ) -> typing.Dict[str, metrics.Stats]:
        return metrics.entity_f1_stats(
            predicted_documents=predictions,
            print_only_tags=None,
            calculate_only_tags=self._resolved_tags,
            min_num_mentions=2,
            ground_truth_documents=ground_truth,
        )

    def train(
        self, train_documents: typing.List[PetDocument]
    ) -> coref.NeuralCoRefSolver:
        solver = coref.NeuralCoRefSolver(
            self._resolved_tags,
            ner_tag_strategy=self._ner_strategy,
            min_mention_overlap=self._mention_overlap,
            min_cluster_overlap=self._cluster_overlap,
        )
        return solver

    @staticmethod
    def predict(
        test_documents: typing.List[PetDocument], estimator: coref.NeuralCoRefSolver
    ) -> typing.List[PetDocument]:
        test_documents = [d.copy(clear=["entities"]) for d in test_documents]
        return estimator.resolve_co_references(test_documents)


class NaiveCoReferenceResolutionStep(PipelineStep):
    def __init__(
        self, name: str, resolved_tags: typing.List[str], mention_overlap: float
    ):
        super().__init__(name)
        self._resolved_tags = resolved_tags
        self._mention_overlap = mention_overlap

    def eval(
        self,
        *,
        predictions: typing.List[PetDocument],
        ground_truth: typing.List[PetDocument],
    ) -> typing.Dict[str, metrics.Stats]:
        return metrics.entity_f1_stats(
            predicted_documents=predictions,
            calculate_only_tags=self._resolved_tags,
            print_only_tags=None,
            min_num_mentions=2,
            ground_truth_documents=ground_truth,
        )

    def train(
        self, train_documents: typing.List[PetDocument]
    ) -> coref.NaiveCoRefSolver:
        solver = coref.NaiveCoRefSolver(
            self._resolved_tags, min_mention_overlap=self._mention_overlap
        )
        return solver

    @staticmethod
    def predict(
        test_documents: typing.List[PetDocument], estimator: coref.NaiveCoRefSolver
    ) -> typing.List[PetDocument]:
        test_documents = [d.copy(clear=["entities"]) for d in test_documents]
        return estimator.resolve_co_references(test_documents)
