import os.path

from augment import base, params
from data import read_documents_from_json
from tests import common


def test_has_variation():
    trafo_classes = common.collect_all_trafos(base.AugmentationStep)
    for clazz in trafo_classes:
        print(f"Testing {clazz.__name__}...")
        doc = read_documents_from_json(os.path.join("tests", "test_docs", "full_doc.json"))[0]
        original_text = " ".join(t.text for t in doc.tokens)
        print(original_text)
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

        aug_is_ok = True

        try:
            augmented = [trafo.do_augment(doc) for _ in range(10)]
            augmented_texts = [" ".join(t.text for t in a.tokens) for a in augmented]
        except:
            print(f"Class {clazz.__name__} failed, skipping...")
            print("-----------")
            print()
            aug_is_ok = False
            continue

        if original_text in augmented_texts:
            print("Augmented texts contain the original text!")
            aug_is_ok = False

        for i in range(len(augmented_texts)):
            a = augmented_texts[i]
            if a in augmented_texts[0:i] + augmented_texts[i+1:]:
                print("Text appears multiple times in augmented texts!")
                print(a)
                aug_is_ok = False
                break

        if aug_is_ok:
            print("Everything fine!")
        print("-----------")
        print()

test_has_variation()