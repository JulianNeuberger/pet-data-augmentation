import dataclasses

import typing


@dataclasses.dataclass
class Document:
    text: str
    name: str
    sentences: typing.List["Sentence"] = dataclasses.field(default_factory=list)
    mentions: typing.List["Mention"] = dataclasses.field(default_factory=list)
    entities: typing.List["Entity"] = dataclasses.field(default_factory=list)
    relations: typing.List["Relation"] = dataclasses.field(default_factory=list)

    def token_offset_for_sentence(self, sentence: int) -> int:
        return sum([s.num_tokens for s in self.sentences[:sentence]])

    def relation_exists_between(
        self, head_entity_index: int, tail_entity_index: int
    ) -> bool:
        for r in self.relations:
            if (
                r.head_entity_index == head_entity_index
                and r.tail_entity_index == tail_entity_index
            ):
                return True
        return False

    def get_relations_by_mention(
        self, mention_index: int, only_head=False, only_tail=False
    ) -> typing.List["Relation"]:
        if only_tail and only_head:
            raise ValueError(
                "The mention can not be only head and tail at the same time!"
            )
        entity_index = self.entity_index_for_mention_index(mention_index)
        ret = []
        for relation in self.relations:
            is_head = relation.head_entity_index == entity_index
            if only_head and is_head:
                ret.append(relation)
                continue

            is_tail = relation.tail_entity_index == entity_index
            if only_tail and is_tail:
                ret.append(relation)
                continue

            if is_tail or is_head:
                ret.append(relation)
        return ret

    def contains_relation(self, relation: "Relation") -> bool:
        return relation.to_tuple(self) in [e.to_tuple(self) for e in self.relations]

    def contains_entity(self, entity: "Entity") -> bool:
        return entity.to_tuple(self) in [e.to_tuple(self) for e in self.entities]

    def entity_index_for_mention_index(self, mention_index: int) -> int:
        for i, e in enumerate(self.entities):
            if mention_index in e.mention_indices:
                return i
        print(mention_index)
        print(self.entities)
        mention = self.mentions[mention_index]
        raise ValueError(
            f"Document contains no entity using mention {mention}, "
            f"which should not happen, but can happen, "
            f"if entities are not properly resolved"
        )

    def mention_index(self, mention: "Mention") -> int:
        mentions_as_tuples = [m.to_tuple(self) for m in self.mentions]
        mention_index = mentions_as_tuples.index(mention.to_tuple(self))
        return mention_index

    def entity_index_for_mention(self, mention: "Mention") -> int:
        mention_index = self.mention_index(mention)
        return self.entity_index_for_mention_index(mention_index)

    def get_mentions_for_token(self, token: "Token") -> typing.List["Mention"]:
        matched = []
        for mention in self.mentions:
            if token.sentence_index != mention.sentence_index:
                continue
            index_in_sentence = token.index_in_sentence(self)
            if index_in_sentence in mention.token_indices:
                matched.append(mention)
        return matched

    def sentence_index_for_token_index(self, token_index: int) -> int:
        assert 0 <= token_index < len(self.tokens)

        tokens_seen = 0
        for sentence_id, sentence in enumerate(self.sentences):
            tokens_seen += len(sentence.tokens)
            if tokens_seen > token_index:
                return sentence_id

    def copy(
        self,
        clear_mentions: bool = False,
        clear_relations: bool = False,
        clear_entities: bool = False,
    ) -> "Document":
        return Document(
            name=self.name,
            text=self.text,
            sentences=[s.copy() for s in self.sentences],
            mentions=[] if clear_mentions else [m.copy() for m in self.mentions],
            relations=[] if clear_relations else [r.copy() for r in self.relations],
            entities=[] if clear_entities else [e.copy() for e in self.entities],
        )

    def merge(self, other: "Document"):
        doc = Document(
            text=self.text + " " + other.text,
            name=self.name + "_" + other.name,
        )

        token_offset = len(self.tokens)
        sentence_offset = len(self.sentences)

        new_sentences = [s.copy() for s in self.sentences]
        for s in other.sentences:
            s = s.copy()
            for t in s.tokens:
                t.index_in_document += token_offset
                t.sentence_index += sentence_offset
            new_sentences.append(s)

        new_mentions = [m.copy() for m in self.mentions]
        for i, m in enumerate(other.mentions):
            m = m.copy()
            m.sentence_index += sentence_offset
            new_mentions.append(m)

        mention_offset = len(self.mentions)

        new_entities = [m.copy() for m in self.entities]
        for e in other.entities:
            e = e.copy()
            e.mention_indices = [i + mention_offset for i in e.mention_indices]
            new_entities.append(e)

        new_relations = [r.copy() for r in self.relations]
        for r in other.relations:
            r = r.copy()
            r.head_entity_index

        new_mentions = self.mentions
        new_mention_ids = {}
        for i, mention in enumerate(other.mentions):
            if mention not in new_mentions:
                new_mention_ids[i] = len(new_mentions)
                new_mentions.append(mention)
            else:
                new_mention_ids[i] = new_mentions.index(mention)

        new_entities = self.entities
        for entity in other.entities:
            mention_indices = [new_mention_ids[i] for i in entity.mention_indices]
            new_entity = PetEntity(mention_indices=tuple(mention_indices))
            if new_entity not in new_entities:
                new_entities.append(new_entity)

        new_relations = self.relations
        for relation in other.relations:
            if relation.head_mention_index not in new_mention_ids:
                continue
            if relation.tail_mention_index not in new_mention_ids:
                continue
            new_relation = PetRelation(
                type=relation.type,
                head_mention_index=new_mention_ids[relation.head_mention_index],
                tail_mention_index=new_mention_ids[relation.tail_mention_index],
            )
            if new_relation not in new_relations:
                new_relations.append(new_relation)

        return PetDocument(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=self.tokens,
            mentions=new_mentions,
            entities=new_entities,
            relations=new_relations,
        )

    @property
    def tokens(self):
        ret = []
        for s in self.sentences:
            ret.extend(s.tokens)
        return ret

    def to_json_serializable(self):
        return {
            "text": self.text,
            "name": self.name,
            "sentences": [s.to_json_serializable() for s in self.sentences],
            "mentions": [m.to_json_serializable() for m in self.mentions],
            "entities": [e.to_json_serializable() for e in self.entities],
            "relations": [r.to_json_serializable() for r in self.relations],
        }


@dataclasses.dataclass
class Sentence:
    tokens: typing.List["Token"] = dataclasses.field(default_factory=list)

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def text(self):
        return " ".join([t.text for t in self.tokens])

    def copy(self) -> "Sentence":
        return Sentence([t.copy() for t in self.tokens])

    def to_json_serializable(self):
        return [t.text for t in self.tokens]


@dataclasses.dataclass
class Mention:
    ner_tag: str
    sentence_index: int
    token_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def document_level_token_indices(self, document: Document) -> typing.List[int]:
        offset = document.token_offset_for_sentence(self.sentence_index)
        return [t + offset for t in self.token_indices]

    def get_tokens(self, document: Document) -> typing.List["Token"]:
        return [
            document.sentences[self.sentence_index].tokens[i]
            for i in self.token_indices
        ]

    def to_tuple(self, *args) -> typing.Tuple:
        return (self.ner_tag.lower(), self.sentence_index) + tuple(self.token_indices)

    def text(self, document: Document):
        return " ".join([t.text for t in self.get_tokens(document)])

    def contains_token(self, token: "Token", document: "Document") -> bool:
        if token.sentence_index != self.sentence_index:
            return False
        for own_token_idx in self.token_indices:
            own_token = document.sentences[self.sentence_index].tokens[own_token_idx]
            if own_token.index_in_document == token.index_in_document:
                return True
        return False

    def pretty_print(self, document: Document):
        return f"{self.text(document)} ({self.ner_tag}, s{self.sentence_index}:{min(self.token_indices)}-{max(self.token_indices)})"

    def copy(self) -> "Mention":
        return Mention(
            ner_tag=self.ner_tag,
            sentence_index=self.sentence_index,
            token_indices=[i for i in self.token_indices],
        )

    def to_json_serializable(self):
        return {
            "tag": self.ner_tag,
            "ner": self.ner_tag,
            "sentence_index": self.sentence_index,
            "sentence_ic": self.sentence_index,
            "token_indices": self.token_indices,
        }


@dataclasses.dataclass
class Entity:
    mention_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def to_tuple(self, document: Document) -> typing.Tuple:
        mentions = [document.mentions[i] for i in self.mention_indices]

        return (frozenset([m.to_tuple(document) for m in mentions]),)

    def copy(self) -> "Entity":
        return Entity(mention_indices=[i for i in self.mention_indices])

    def get_tag(self, document: Document) -> str:
        tags = list(set([document.mentions[m].ner_tag for m in self.mention_indices]))
        assert len(tags) == 1
        return tags[0]

    def pretty_print(self, document: Document):
        return f"Entity [{[document.mentions[m].pretty_print(document) for m in self.mention_indices]}]"

    def to_json_serializable(self):
        return {"mention_indices": self.mention_indices}


@dataclasses.dataclass
class Relation:
    head_entity_index: int
    tail_entity_index: int
    tag: str
    evidence: typing.List[int]

    def copy(self) -> "Relation":
        return Relation(
            head_entity_index=self.head_entity_index,
            tail_entity_index=self.tail_entity_index,
            tag=self.tag,
            evidence=[i for i in self.evidence],
        )

    def to_tuple(self, document: Document) -> typing.Tuple:
        return (
            self.tag.lower(),
            document.entities[self.head_entity_index].to_tuple(document),
            document.entities[self.tail_entity_index].to_tuple(document),
        )

    def pretty_print(self, document: Document):
        head_entity = document.entities[self.head_entity_index]
        tail_entity = document.entities[self.tail_entity_index]

        head_mention = document.mentions[head_entity.mention_indices[0]]
        tail_mention = document.mentions[tail_entity.mention_indices[0]]

        return (
            f"[{head_mention.pretty_print(document)} (+{len(head_entity.mention_indices) - 1})]"
            f"--[{self.tag}]-->"
            f"[{tail_mention.pretty_print(document)} (+{len(tail_entity.mention_indices) - 1})]"
        )

    def to_json_serializable(self):
        return {
            "head": self.head_entity_index,
            "tail": self.tail_entity_index,
            "tag": self.tag,
            "type": self.tag,
            "evidence": self.evidence,
        }


@dataclasses.dataclass
class Token:
    text: str
    index_in_document: int
    pos_tag: str
    bio_tag: str
    sentence_index: int

    def index_in_sentence(self, doc: "Document") -> int:
        sentence = doc.sentences[self.sentence_index]
        for index_in_sentence, token in enumerate(sentence.tokens):
            if token.index_in_document == self.index_in_document:
                return index_in_sentence
        raise IndexError(
            f"Could not find token in sentence with id {self.sentence_index}."
        )

    def copy(self) -> "Token":
        return Token(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            bio_tag=self.bio_tag,
            sentence_index=self.sentence_index,
        )
