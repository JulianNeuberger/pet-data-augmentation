from augment import trafo52


def test_load():
    abbreviations = trafo52.Trafo52Step._load()

    assert len(abbreviations) > 0
