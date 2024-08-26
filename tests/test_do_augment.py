

from augment import base, params
from tests.common import collect_all_trafos, document_fixture


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
