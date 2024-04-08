import inspect
import typing

from augment import base, params
from data import model


def collect_all_trafos(
    base_class: typing.Type,
) -> typing.List[typing.Type[base.AugmentationStep]]:
    sub_classes = []

    immediate_sub_classes = base_class.__subclasses__()
    sub_classes.extend([c for c in immediate_sub_classes if not inspect.isabstract(c)])
    for sub_class in immediate_sub_classes:
        child_sub_classes = collect_all_trafos(sub_class)
        sub_classes.extend(child_sub_classes)

    return sub_classes


def document_fixture():
    doc = model.Document(
        text="",
        name="test",
        sentences=[
            model.Sentence(
                tokens=[
                    model.Token("This", 0, "A", "O", 0),
                    model.Token("is", 1, "B", "O", 0),
                    model.Token("a", 2, "C", "O", 0),
                    model.Token("sentence", 3, "A", "B-Object", 0),
                    model.Token(".", 4, "A", "O", 0),
                ]
            ),
            model.Sentence(
                tokens=[
                    model.Token("And", 5, "A", "O", 1),
                    model.Token("another", 6, "B", "B-Object", 1),
                    model.Token("one", 7, "C", "I-Object", 1),
                    model.Token("!", 8, "A", "0", 1),
                ]
            ),
            model.Sentence(
                tokens=[
                    model.Token("Here", 9, "A", "B-Something", 2),
                    model.Token("comes", 10, "B", "B-Activity", 2),
                    model.Token("the", 11, "C", "O", 2),
                    model.Token("last", 12, "C", "O", 2),
                    model.Token("one", 13, "C", "B-Object", 2),
                    model.Token(".", 14, "A", "0", 2),
                ]
            ),
        ],
        mentions=[
            model.Mention("Object", 0, [3]),
            model.Mention("Object", 1, [1, 2]),
            model.Mention("Something", 2, [0]),
            model.Mention("Activity", 2, [1]),
            model.Mention("Object", 2, [4]),
        ],
        entities=[
            model.Entity([0, 1]),
            model.Entity([2]),
            model.Entity([3]),
            model.Entity([4]),
        ],
        relations=[
            model.Relation(0, 2, "Testtag", [0, 2]),
            model.Relation(1, 2, "Othertag", [1, 2]),
            model.Relation(3, 2, "Lasttag", [2]),
        ],
    )
    return doc


def test_do_augment():
    trafo_classes = collect_all_trafos(base.AugmentationStep)
    print(trafo_classes)
    for clazz in trafo_classes:
        print(f"Testing {clazz.__name__}...")
        doc = document_fixture()
        args = {"dataset": [doc]}
        for param in clazz.get_params():
            if isinstance(param, params.NumberParam):
                args[param.name] = param.max_value
            elif isinstance(param, params.ChoiceParam):
                args[param.name] = param.choices[0]
                if param.max_num_picks > 1:
                    args[param.name] = [param.choices[0]]
            elif isinstance(param, params.BooleanParameter):
                args[param.name] = True
        trafo = clazz(**args)
        augmented = trafo.do_augment(doc)
