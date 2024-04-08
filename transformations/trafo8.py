import copy

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from data import model

# Author: Benedikt
# implementation Failed, due to the Generation Model
class BackTranslation():
    def __init__(self, max_outputs=1, num_beams=2):
        self.max_outputs = max_outputs
        self.num_beams = num_beams
        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(
            name_en_de
        )
        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(
            name_de_en
        )

    def back_translate(self, en: str):
        try:
            de = self.en2de(en)
            en_new = self.de2en(de)
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(
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
        decoded = self.tokenizer_en_de.decode(
            outputs[0], skip_special_tokens=True
        )
        return decoded

    def generate(self, doc: model.Document):
        for sentence in doc.sentences:
            for token in sentence.tokens:
                text = copy.deepcopy(token.text)
                translated_text = self.back_translate(token.text)
                print(translated_text)

tokens = [model.Token(text="I leave", index_in_document=0,
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

trafo = BackTranslation()
trafo.generate(doc)
#print(trafo.de2en(""))