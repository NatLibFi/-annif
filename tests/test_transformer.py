"""Unit tests for the input-transformers in Annif"""

import pytest
import annif.transformer
from annif.exception import ConfigurationException
from annif.transformer import parse_specs


def test_parse_specs():
    parsed = parse_specs('foo, bar(42,43,key=abc)')
    assert parsed == [('foo', [], {}), ('bar', ['42', '43'], {'key': 'abc'})]


def test_get_transform_nonexistent():
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transform("nonexistent", project=None)


def test_get_transform_badspec(project):
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transform("pass(invalid_argument)", project)


def test_input_limiter():
    transf = annif.transformer.get_transform("limit(3)", project=None)
    assert transf.transform_text("running") == "run"


def test_input_limiter_with_negative_value(project):
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transform("limit(-2)", project)


def test_lang_filter(project):
    transf = annif.transformer.get_transform("filter_lang", project)
    text = """
        Kansalliskirjasto on kaikille avoin kulttuuriperintöorganisaatio, joka
        palvelee valtakunnallisesti kansalaisia, tiedeyhteisöjä ja muita
        yhteiskunnan toimijoita.
        The National Library of Finland is the oldest and largest scholarly
        library in Finland. It is responsible for the collection, description,
        preservation and accessibility of Finland’s published national heritage
        and the unique collections under its care.
        Nationalbiblioteket är Finlands största och äldsta vetenskapliga
        bibliotek, som ansvarar för utökning, beskrivning, förvaring och
        tillhandahållande av vårt nationella publikationsarv och av sina unika
        samlingar.
        Turvaamme Suomessa julkaistun tai Suomea koskevan julkaistun
        kulttuuriperinnön saatavuuden sekä välittämme ja tuotamme
        tietosisältöjä tutkimukselle, opiskelulle, kansalaisille ja
        yhteiskunnalle. Kehitämme palveluja yhteistyössä kirjastojen,
        arkistojen, museoiden ja muiden toimijoiden kanssa.
    """
    text = ' '.join(text.split())
    text_filtered = """
        Kansalliskirjasto on kaikille avoin kulttuuriperintöorganisaatio, joka
        palvelee valtakunnallisesti kansalaisia, tiedeyhteisöjä ja muita
        yhteiskunnan toimijoita.
        Turvaamme Suomessa julkaistun tai Suomea koskevan julkaistun
        kulttuuriperinnön saatavuuden sekä välittämme ja tuotamme
        tietosisältöjä tutkimukselle, opiskelulle, kansalaisille ja
        yhteiskunnalle. Kehitämme palveluja yhteistyössä kirjastojen,
        arkistojen, museoiden ja muiden toimijoiden kanssa.
    """
    text_filtered = ' '.join(text_filtered.split())
    assert transf.transform_text(text) == text_filtered


def test_chained_transforms_text():
    transf = annif.transformer.get_transform(
        "limit(5),pass,limit(3),", project=None)
    assert transf.transform_text("abcdefghij") == "abc"

    # Check with a more arbitrary transform function
    reverser = annif.transformer.transformer.IdentityTransform(None)
    reverser.transform_fn = lambda x: x[::-1]
    transf.transforms.append(reverser)
    assert transf.transform_text("abcdefghij") == "cba"


def test_chained_transforms_corpus(document_corpus):
    transf = annif.transformer.get_transform(
        "limit(5),pass,limit(3),", project=None)
    transformed_corpus = transf.transform_corpus(document_corpus)
    for transf_doc, doc in zip(transformed_corpus.documents,
                               document_corpus.documents):
        assert transf_doc.text == doc.text[:3]
        assert transf_doc.uris == doc.uris
        assert transf_doc.labels == doc.labels

    # Check with a more arbitrary transform function
    reverser = annif.transformer.transformer.IdentityTransform(None)
    reverser.transform_fn = lambda x: x[::-1]
    transf.transforms.append(reverser)
    for transf_doc, doc in zip(transformed_corpus.documents,
                               document_corpus.documents):
        assert transf_doc.text == doc.text[:3][::-1]
        assert transf_doc.uris == doc.uris
        assert transf_doc.labels == doc.labels
