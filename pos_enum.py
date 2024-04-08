from enum import Enum


class Pos(Enum):
    VERB = 1
    NOUN = 2
    AD = 3
    PRON = 4

    @property
    def tags(self):
        if self.value == 1:
            return ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        elif self.value == 2:
            return ["NNS", "NN", "NNP", "NNPS"]
        elif self.value == 3:
            return ["RB", "RBS", "RBR", "JJ", "JJR", "JJS"]
        else:
            return ["PRP", "PRP$", "WP", "WP$"]
