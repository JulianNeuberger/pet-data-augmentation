from data import model
import typing
import data
import matplotlib.pyplot as pp
from collections import Counter

docs: typing.List[model.Document] = data.loader.read_documents_from_json('../complete.json')

# Methods for Analysing the PET Dataset


# Author: Benedikt
# for plotting
def create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name):
    count_array = []
    count_array2 = []
    count_array.append(actor_count)
    count_array.append(activity_count)
    count_array.append(activity_data_count)
    count_array2.append(further_count)
    count_array.append(xor_count)
    count_array.append(cond_count)
    count_array2.append(and_count)
    label = ["Actor", "Activity", "Acitivity Data", "XOR", "Cond"]
    label2 = ["Further Spec", "AND"]
    figure, axes = pp.subplots(figsize=(10, 5))
    figure.suptitle(f'{name} Mention')
    axes.set_xlabel("Entitäten")
    axes.set_ylabel("Anzahl")
    axes.bar(x=label, height=count_array, color="darkgrey", edgecolor="black", width=0.5)
    axes.bar(x=label2, height=count_array2, color="orange", edgecolor="black", width=0.5)
    pp.savefig(f"Analysis/{name}.png")
    pp.show()


# Author: Leonie
# for plotting
def create_graph_relation(dict, name):
    figure, axes = pp.subplots(figsize=(10, 5))
    figure.suptitle(name)
    axes.set_xlabel("POS TAGS")
    axes.set_ylabel("Anzahl")
    names = list(dict.keys())
    values = list(dict.values())
    axes.bar(range((len(dict))), values, tick_label=names, color="darkgrey", edgecolor="black", width=0.5)
    #pp.savefig(f"AnalysisRelation/{name}.png")
    pp.show()


# every Method counts the entity that stands behind count_...
# Author: Leonie
def count_nomen_furtherspec():
    counter = 0
    for doc in docs:
        for entity in doc.entities:
            if entity.get_tag(doc) == "Further Specification":
                for i in entity.mention_indices:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        print(token.pos_tag)
                        if token.pos_tag == "NN":
                            counter += 1
                    text = mention.text(doc)
                    print(text)

            print('-' * 10)
    print(f'Counter: {counter}')


# Author: Leonie
def count_adverbs():
    count = 0
    for doc in docs:
        for mention in doc.mentions:
            for token in mention.get_tokens(doc):
                if token.pos_tag == "RB" or token.pos_tag == "RBR" or token.pos_tag == "RBS":
                    count += 1

    return count


# Author: Benedikt
def count_all():
    actor_count = 0
    activity_count = 0
    activity_data_count = 0
    further_count = 0
    xor_count = 0
    cond_count = 0
    and_count = 0
    total_count = 0
    name = "Alle"

    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):

                    total_count += 1
                    if entity.get_tag(doc) == "Actor":
                        actor_count += 1
                    if entity.get_tag(doc) == "Activity":
                        activity_count += 1
                    if entity.get_tag(doc) == "Activity Data":
                        activity_data_count += 1
                    if entity.get_tag(doc) == "Further Specification":
                        further_count += 1
                    if entity.get_tag(doc) == "XOR Gateway":
                        xor_count += 1
                    if entity.get_tag(doc) == "Condition Specification":
                        cond_count += 1
                    if entity.get_tag(doc) == "AND Gateway":
                        and_count += 1

    print(f'Actor Count: {actor_count} {(actor_count / total_count) * 100}')
    print(f'Activity Count: {activity_count} {(activity_count / total_count) * 100}')
    print(f'Activity Data Count: {activity_data_count} {(activity_data_count / total_count) * 100}')
    print(f'Further Spec Count: {further_count} {(further_count / total_count) * 100}')
    print(f'XOR Count: {xor_count} {(xor_count / total_count) * 100}')
    print(f'Cond Count: {cond_count} {(cond_count / total_count) * 100}')
    print(f'AND Count: {and_count} {(and_count / total_count) * 100}')
    print(f'Total: {total_count}')
    print(f'Gesamtanteil:  {(total_count / 4126) * 100}')
    print('_' * 20)
    create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name)


# Author: Benedikt
def count_nomen():
    actor_count = 0
    activity_count = 0
    activity_data_count = 0
    further_count = 0
    xor_count = 0
    cond_count = 0
    and_count = 0
    total_count = 0
    name = "nomen"
    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "NN" or token.pos_tag == "NNS":
                        total_count += 1
                        if entity.get_tag(doc) == "Actor":
                            actor_count += 1
                        if entity.get_tag(doc) == "Activity":
                            activity_count += 1
                        if entity.get_tag(doc) == "Activity Data":
                            activity_data_count += 1
                        if entity.get_tag(doc) == "Further Specification":
                            further_count += 1
                        if entity.get_tag(doc) == "XOR Gateway":
                            xor_count += 1
                        if entity.get_tag(doc) == "Condition Specification":
                            cond_count += 1
                        if entity.get_tag(doc) == "AND Gateway":
                            and_count += 1

    print(f'Actor Count: {actor_count} {(actor_count / total_count) * 100}')
    print(f'Activity Count: {activity_count} {(activity_count / total_count) * 100}')
    print(f'Activity Data Count: {activity_data_count} {(activity_data_count / total_count) * 100}')
    print(f'Further Spec Count: {further_count} {(further_count / total_count) * 100}')
    print(f'XOR Count: {xor_count} {(xor_count / total_count) * 100}')
    print(f'Cond Count: {cond_count} {(cond_count / total_count) * 100}')
    print(f'AND Count: {and_count} {(and_count / total_count) * 100}')
    print(f'Total: {total_count}')
    print(f'Gesamtanteil:  {(total_count / 4126) * 100}')
    print('_' * 20)
    create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name)


# Author: Leonie
def count_nomen_p():
    actor_count = 0
    activity_count = 0
    activity_data_count = 0
    further_count = 0
    xor_count = 0
    cond_count = 0
    and_count = 0
    total_count = 0
    name = "eigennamen"
    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "NNP" or token.pos_tag == "NNPS":
                        total_count += 1
                        if entity.get_tag(doc) == "Actor":
                            actor_count += 1
                        if entity.get_tag(doc) == "Activity":
                            activity_count += 1
                        if entity.get_tag(doc) == "Activity Data":
                            activity_data_count += 1
                        if entity.get_tag(doc) == "Further Specification":
                            further_count += 1
                        if entity.get_tag(doc) == "XOR Gateway":
                            xor_count += 1
                        if entity.get_tag(doc) == "Condition Specification":
                            cond_count += 1
                        if entity.get_tag(doc) == "AND Gateway":
                            and_count += 1

    print(f'Actor Count: {actor_count} {(actor_count / total_count) * 100}')
    print(f'Activity Count: {activity_count} {(activity_count / total_count) * 100}')
    print(f'Activity Data Count: {activity_data_count} {(activity_data_count / total_count) * 100}')
    print(f'Further Spec Count: {further_count} {(further_count / total_count) * 100}')
    print(f'XOR Count: {xor_count} {(xor_count / total_count) * 100}')
    print(f'Cond Count: {cond_count} {(cond_count / total_count) * 100}')
    print(f'AND Count: {and_count} {(and_count / total_count) * 100}')
    print(f'Total: {total_count}')
    print(f'Gesamtanteil:  {(total_count / 4126) * 100}')
    print('_' * 20)
    create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name)


# Author: Leonie
def count_adj():
    actor_count = 0
    activity_count = 0
    activity_data_count = 0
    further_count = 0
    xor_count = 0
    cond_count = 0
    and_count = 0
    total_count = 0
    name = "Adj"
    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "JJ" or token.pos_tag == "JJR" or token.pos_tag == "JJS":
                        total_count += 1
                        if entity.get_tag(doc) == "Actor":
                            actor_count += 1
                        if entity.get_tag(doc) == "Activity":
                            activity_count += 1
                        if entity.get_tag(doc) == "Activity Data":
                            activity_data_count += 1
                        if entity.get_tag(doc) == "Further Specification":
                            further_count += 1
                        if entity.get_tag(doc) == "XOR Gateway":
                            xor_count += 1
                        if entity.get_tag(doc) == "Condition Specification":
                            cond_count += 1
                        if entity.get_tag(doc) == "AND Gateway":
                            and_count += 1

    print(f'Actor Count: {actor_count} {(actor_count / total_count) * 100}')
    print(f'Activity Count: {activity_count} {(activity_count / total_count) * 100}')
    print(f'Activity Data Count: {activity_data_count} {(activity_data_count / total_count) * 100}')
    print(f'Further Spec Count: {further_count} {(further_count / total_count) * 100}')
    print(f'XOR Count: {xor_count} {(xor_count / total_count) * 100}')
    print(f'Cond Count: {cond_count} {(cond_count / total_count) * 100}')
    print(f'AND Count: {and_count} {(and_count / total_count) * 100}')
    print(f'Total: {total_count}')
    print(f'Gesamtanteil:  {(total_count / 4126) * 100}')
    print('_' * 20)
    create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name)


# Author: Benedikt
def count_adv():
    actor_count = 0
    activity_count = 0
    activity_data_count = 0
    further_count = 0
    xor_count = 0
    cond_count = 0
    and_count = 0
    total_count = 0
    name = "adv"
    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "RB" or token.pos_tag == "RBR" or token.pos_tag == "RBS":
                        total_count += 1
                        if entity.get_tag(doc) == "Actor":
                            actor_count += 1
                        if entity.get_tag(doc) == "Activity":
                            activity_count += 1
                        if entity.get_tag(doc) == "Activity Data":
                            activity_data_count += 1
                        if entity.get_tag(doc) == "Further Specification":
                            further_count += 1
                        if entity.get_tag(doc) == "XOR Gateway":
                            xor_count += 1
                        if entity.get_tag(doc) == "Condition Specification":
                            cond_count += 1
                        if entity.get_tag(doc) == "AND Gateway":
                            and_count += 1

    print(f'Actor Count: {actor_count} {(actor_count / total_count) * 100}')
    print(f'Activity Count: {activity_count} {(activity_count / total_count) * 100}')
    print(f'Activity Data Count: {activity_data_count} {(activity_data_count / total_count) * 100}')
    print(f'Further Spec Count: {further_count} {(further_count / total_count) * 100}')
    print(f'XOR Count: {xor_count} {(xor_count / total_count) * 100}')
    print(f'Cond Count: {cond_count} {(cond_count / total_count) * 100}')
    print(f'AND Count: {and_count} {(and_count / total_count) * 100}')
    print(f'Total: {total_count}')
    print(f'Gesamtanteil:  {(total_count / 4126) * 100}')
    print('_' * 20)
    create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name)


# Author: Leonie
def count_verb():
    actor_count = 0
    activity_count = 0
    activity_data_count = 0
    further_count = 0
    xor_count = 0
    cond_count = 0
    and_count = 0
    total_count = 0
    name = "verb"
    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "VB" or token.pos_tag == "VBD" or token.pos_tag == "VBG" or token.pos_tag == "VBN" or token.pos_tag == "VBP" or token.pos_tag == "VBZ":
                        total_count += 1
                        if entity.get_tag(doc) == "Actor":
                            actor_count += 1
                        if entity.get_tag(doc) == "Activity":
                            activity_count += 1
                        if entity.get_tag(doc) == "Activity Data":
                            activity_data_count += 1
                        if entity.get_tag(doc) == "Further Specification":
                            further_count += 1
                        if entity.get_tag(doc) == "XOR Gateway":
                            xor_count += 1
                        if entity.get_tag(doc) == "Condition Specification":
                            cond_count += 1
                        if entity.get_tag(doc) == "AND Gateway":
                            and_count += 1

    print(f'Actor Count: {actor_count} {(actor_count / total_count) * 100}')
    print(f'Activity Count: {activity_count} {(activity_count / total_count) * 100}')
    print(f'Activity Data Count: {activity_data_count} {(activity_data_count / total_count) * 100}')
    print(f'Further Spec Count: {further_count} {(further_count / total_count) * 100}')
    print(f'XOR Count: {xor_count} {(xor_count / total_count) * 100}')
    print(f'Cond Count: {cond_count} {(cond_count / total_count) * 100}')
    print(f'AND Count: {and_count} {(and_count / total_count) * 100}')
    print(f'Total: {total_count}')
    print(f'Gesamtanteil:  {(total_count / 4126) * 100}')
    print('_' * 20)
    create_graph(actor_count, activity_count, activity_data_count, further_count, xor_count, cond_count, and_count,
                 name)


# Author: Benedikt
def count_entities():
    pos_tag_list_actor = []
    pos_tag_list_activity = []
    pos_tag_list_activitydata = []
    pos_tag_list_xor = []
    pos_tag_list_cond = []
    pos_tag_list_and = []
    pos_tag_list_further = []
    pos_tag_list_all = []
    for doc in docs:
        for entity in doc.entities:

            for i in entity.mention_indices:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    pos_tag_list_all.append(token.pos_tag)
                    if mention.ner_tag == "Actor":
                        pos_tag_list_actor.append(token.pos_tag)
                    if mention.ner_tag == "Activity":
                        pos_tag_list_activity.append(token.pos_tag)
                    if mention.ner_tag == "Activity Data":
                        pos_tag_list_activitydata.append(token.pos_tag)
                    if mention.ner_tag == "XOR Gateway":
                        pos_tag_list_xor.append(token.pos_tag)
                    if mention.ner_tag == "Condition Specification":
                        pos_tag_list_cond.append(token.pos_tag)
                    if mention.ner_tag == "AND Gateway":
                        pos_tag_list_and.append(token.pos_tag)
                    if mention.ner_tag == "Further Specification":
                        pos_tag_list_further.append(token.pos_tag)

    #create_graph_relation(Counter(pos_tag_list_actor), "Actor")
    #create_graph_relation(Counter(pos_tag_list_activity), "Activity")
    #create_graph_relation(Counter(pos_tag_list_activitydata), "Activity Data")
    #create_graph_relation(Counter(pos_tag_list_xor), "XOR Gateway")
    #create_graph_relation(Counter(pos_tag_list_cond), "Condition Specification")
    #create_graph_relation(Counter(pos_tag_list_and), "AND Gateway")
    print(Counter(pos_tag_list_xor))
    print(len(pos_tag_list_xor))
    #create_graph_relation(Counter(pos_tag_list_further), "Further Specification")
    #create_graph_relation(Counter(pos_tag_list_all), "All Entities")+


# Author: Leonie
def count_relations():
    pos_tag_list_same = []
    pos_tag_list_further = []
    pos_tag_list_actper = []
    pos_tag_list_flow = []
    pos_tag_list_actrec = []
    pos_tag_list_uses = []
    pos_tag_list_all = []
    for doc in docs:
        entities_as_tuples = [frozenset(e.mention_indices) for e in doc.entities]

        for relation in doc.relations:

            if relation.tag == "same gateway":
                for i in entities_as_tuples[relation.tail_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_same.append(token.pos_tag)
                for i in entities_as_tuples[relation.head_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_same.append(token.pos_tag)

            if relation.tag == "further specification":
                for i in entities_as_tuples[relation.tail_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_further.append(token.pos_tag)
                for i in entities_as_tuples[relation.head_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_further.append(token.pos_tag)

            if relation.tag == "actor performer":
                for i in entities_as_tuples[relation.tail_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_actper.append(token.pos_tag)
                for i in entities_as_tuples[relation.head_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_actper.append(token.pos_tag)
            if relation.tag == "flow":
                for i in entities_as_tuples[relation.tail_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_flow.append(token.pos_tag)
                for i in entities_as_tuples[relation.head_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_flow.append(token.pos_tag)
            if relation.tag == "actor recipient":
                for i in entities_as_tuples[relation.tail_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_actrec.append(token.pos_tag)
                for i in entities_as_tuples[relation.head_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_actrec.append(token.pos_tag)
            if relation.tag == "uses":
                for i in entities_as_tuples[relation.tail_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_uses.append(token.pos_tag)
                for i in entities_as_tuples[relation.head_entity_index]:
                    mention = doc.mentions[i]
                    for token in mention.get_tokens(doc):
                        pos_tag_list_uses.append(token.pos_tag)
            for i in entities_as_tuples[relation.tail_entity_index]:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    pos_tag_list_all.append(token.pos_tag)
            for i in entities_as_tuples[relation.head_entity_index]:
                mention = doc.mentions[i]
                for token in mention.get_tokens(doc):
                    pos_tag_list_all.append(token.pos_tag)

    #create_graph_relation(Counter(pos_tag_list_same), "same_Gateway")
    #create_graph_relation(Counter(pos_tag_list_further), "Further_Specification")
    #create_graph_relation(Counter(pos_tag_list_actper), "Actor_Performer")
    #create_graph_relation(Counter(pos_tag_list_flow), "Flow")
    #create_graph_relation(Counter(pos_tag_list_actrec), "Actor Recipient")
    #create_graph_relation(Counter(pos_tag_list_uses), "Uses")
    create_graph_relation(Counter(pos_tag_list_all), "All Relations")


