from augment import trafo82


def test_load():
    abbreviations = trafo82.Trafo82Step._load()

    assert len(abbreviations) > 0
