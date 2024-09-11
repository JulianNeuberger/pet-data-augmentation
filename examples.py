import json
import os
import pathlib
import typing

import augment
import data
from augment import params


def get_param_value_json(p: augment.params.Param):
    return input(f"{p.name}=")


data_path = (pathlib.Path(__file__).parent / "jsonl" / "all.new.jsonl").resolve()
all_documents = data.pet.NewPetFormatImporter(str(data_path)).do_import()

# strategies: typing.List[typing.Type[augment.AugmentationStep]] = (
#     augment.collect_all_augmentations(augment.base.AugmentationStep)
# )
strategies: typing.List[typing.Type[augment.AugmentationStep]] = [augment.RandomInsert]

examples_path = (pathlib.Path(__file__).parent / "res" / "examples").resolve()
os.makedirs(examples_path, exist_ok=True)

example_document = all_documents.pop(1)
dataset = all_documents

for strategy_class in strategies:
    print("#" * 250)
    print(f"### {strategy_class.__name__}")
    print("#" * 250)
    print()
    param_values = {}
    for param in strategy_class.get_params():
        print(param)
        while True:
            param_value_json = get_param_value_json(param)

            try:
                choice = json.loads(param_value_json)
                if isinstance(param, params.ChoiceParam):
                    if param.max_num_picks > 1:
                        value = [param.choices[i] for i in choice]
                    else:
                        value = param.choices[choice]
                    param_values[param.name] = value
                else:
                    param_values[param.name] = json.loads(param_value_json)
            except json.decoder.JSONDecodeError:
                print("Invalid value.")
                continue
            break
    strategy = strategy_class(dataset, **param_values)
    augmented = strategy.do_augment(example_document, num_augments=10)
    print()
    print(f"Generated {len(augmented)} augmented documents!")

    target_file = examples_path / f"{strategy_class.__name__}.jsonl"
    data.pet.PetJsonExporter(str(target_file)).export(augmented)

    print()
    print("-" * 250)
    print()
    print()
