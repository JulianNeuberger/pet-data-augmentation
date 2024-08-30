from augment import base, params, collect_all_augmentations
from tests.common import document_fixture


def test_do_augment():
    trafo_classes = collect_all_augmentations(base.AugmentationStep)
    print(trafo_classes)
    for clazz in trafo_classes:
        print(f"Testing {clazz.__name__}...")
        doc = document_fixture()
        dataset = [doc.copy(clear=[]), doc.copy(clear=[]).merge(doc.copy(clear=[]))]
        test_dataset = [doc.copy(clear=[])]
        args = {}
        for param in clazz.get_params():
            if isinstance(param, params.NumberParam):
                args[param.name] = param.max_value
            elif isinstance(param, params.ChoiceParam):
                args[param.name] = param.choices[0]
                if param.max_num_picks > 1:
                    args[param.name] = [param.choices[0]]
            elif isinstance(param, params.BooleanParameter):
                args[param.name] = True
        trafo = clazz(dataset, **args)
        augmented = trafo.do_augment(doc, num_augments=5)
