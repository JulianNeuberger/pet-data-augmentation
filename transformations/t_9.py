import itertools

from data import model
import typing
import data
import numpy as np
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

docs: typing.List[model.Document] = data.loader.read_documents_from_json('../../one_doc.json')

# Author: Benedikt
# implementation Failed, due to the Generation Model, same error as the one for Transformation 8
def generate(doc,p=0.5, segment_length=5):

    augmentations = []
    for sentence in doc.sentences:
        token_seq = []
        tag_seq = []
        for token in sentence.tokens:
            token_seq.append(token.text)
            tag_seq.append(token.bio_tag)
        segment_tokens, segment_tags = create_segments(token_seq, tag_seq)

        tokens, tags = [], []
        for s_token, s_tag in zip(segment_tokens, segment_tags):
            translate_segment = np.random.binomial(1, p)
            translate_segment = True
            if (
                    s_tag[0] != "O"
                    or len(s_token) < segment_length
                    or not translate_segment):
                tokens.extend(s_token)
                tags.extend(s_tag)
                continue
            segment_text = " ".join(s_token)
            segment_translation = translation_pipeline(segment_text)
            translated_tokens = segment_translation.split()
            translated_tags = ["O"] * len(translated_tokens)
            tokens.extend(translated_tokens)
            tags.extend(translated_tags)
        augmentations.append(([tokens, tags]))
    print(augmentations)


def create_segments(tokens, tags):
    """
    A segment is defined as a consecutive sequence of same tag/label.
    """

    segment_tokens, segment_tags = [], []
    tags_idxs = [(i, t) for i, t in enumerate(tags)]
    groups = [
        list(g)
        for _, g in itertools.groupby(
            tags_idxs, lambda s: s[1].split("-")[-1]
        )
    ]
    for group in groups:
        idxs = [i[0] for i in group]
        segment_tokens.append([tokens[idx] for idx in idxs])
        segment_tags.append([tags[idx] for idx in idxs])

    return segment_tokens, segment_tags

def translation_pipeline(text):
    """
    Pass the text in source languages through the intermediate
    translations.
    :param text: the text to translate
    :return text_trans: backtranslated text
    """
    mname_en2de = "facebook/wmt19-en-de"
    mname_de2en = "facebook/wmt19-de-en"
    tokenizer_en2de = FSMTTokenizer.from_pretrained(mname_en2de)
    tokenizer_de2en = FSMTTokenizer.from_pretrained(mname_de2en)
    model_en2de = FSMTForConditionalGeneration.from_pretrained(
        mname_en2de
    )
    model_de2en = FSMTForConditionalGeneration.from_pretrained(
        mname_de2en
    )
    en2de_inputids = tokenizer_en2de.encode(text, return_tensors="pt")
    outputs_en2de = model_en2de.generate(en2de_inputids)
    text_trans = tokenizer_en2de.decode(
        outputs_en2de[0], skip_special_tokens=True
    )
    de2en_inputids = tokenizer_de2en.encode(
        text_trans, return_tensors="pt"
    )
    outputs_de2en = model_de2en.generate(de2en_inputids)
    text_trans = tokenizer_en2de.decode(
        outputs_de2en[0], skip_special_tokens=True
    )
    return text_trans

def back_translate(en: str):
    try:
        de = en2de(en)
        en_new = de2en(de)
    except Exception:
        print("Returning Default due to Run Time Exception")
        en_new = en
    return en_new

def en2de(input):
    input_ids = FSMTTokenizer.tokenizer_en_de.encode(input, return_tensors="pt")
    outputs = FSMTForConditionalGeneration.model_en_de.generate(input_ids)
    decoded = FSMTTokenizer.tokenizer_en_de.decode(
        outputs[0], skip_special_tokens=True
    )
    return decoded

def de2en(self, input):
    input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
    outputs = self.model_de_en.generate(
        input_ids,
        num_return_sequences=self.max_outputs,
        num_beams=self.num_beams,
    )
    predicted_outputs = []
    for output in outputs:
        decoded = self.tokenizer_de_en.decode(
            output, skip_special_tokens=True
        )
        predicted_outputs.append(decoded)
    if self.verbose:
        print(predicted_outputs)  # Machine learning is great, isn't it?
    return predicted_outputs

tokens = [model.Token(text="I", index_in_document=0,
                          pos_tag="PRP", bio_tag="B-Actor",
                          sentence_index=0), model.Token(text="leave", index_in_document=1,
                                                         pos_tag="VBP", bio_tag="O",
                                                         sentence_index=0), model.Token(text="Human", index_in_document=2,
                                                                                        pos_tag="NN", bio_tag="B-Activity Data",
                                                                                        sentence_index=0),
              model.Token(text="resources", index_in_document=3,
                          pos_tag="NNS", bio_tag="I-Activity Data",
                          sentence_index=0),
              model.Token(text="office", index_in_document=4,
                          pos_tag="NN", bio_tag="I-Activity Data",
                          sentence_index=0),
              model.Token(text="deadline", index_in_document=5,
                          pos_tag="NN", bio_tag="I-Activity Data",
                          sentence_index=0)]

sentence1 = model.Sentence(tokens=tokens)

doc = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence1],
    mentions=[],
    entities=[],
    relations=[])
#generate(doc)
print(translation_pipeline(""))