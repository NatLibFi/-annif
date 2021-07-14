from annif.backend import get_backend
import annif.corpus
from annif.backend.stwfsa import StwfsaBackend
from annif.exception import NotInitializedException, NotSupportedException
import pytest


_backend_conf = {
    'language': 'fi',
    'concept_type_uri': 'http://www.w3.org/2004/02/skos/core#Concept',
    'sub_thesaurus_type_uri':
        'http://www.w3.org/2004/02/skos/core#Collection',
    'thesaurus_relation_type_uri':
        'http://www.w3.org/2004/02/skos/core#member',
    'thesaurus_relation_is_specialisation': True,
}


def test_stwfsa_default_params(project):
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id=StwfsaBackend.name,
        config_params={},
        project=project
    )
    expected_default_params = {
        'concept_type_uri': 'http://www.w3.org/2004/02/skos/core#Concept',
        'sub_thesaurus_type_uri':
            'http://www.w3.org/2004/02/skos/core#Collection',
        'thesaurus_relation_type_uri':
            'http://www.w3.org/2004/02/skos/core#member',
        'thesaurus_relation_is_specialisation': True,
        'remove_deprecated': True,
        'handle_title_case': True,
        'extract_upper_case_from_braces': True,
        'extract_any_case_from_braces': False,
        'expand_ampersand_with_spaces': True,
        'expand_abbreviation_with_punctuation': True,
        'simple_english_plural_rules': False,
        'use_txt_vec': False,
    }
    actual_params = stwfsa.params
    assert expected_default_params == actual_params


def test_stwfsa_not_initialized(project):
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id='stwfsa',
        config_params={},
        project=project)
    with pytest.raises(NotInitializedException):
        stwfsa.suggest("example text")


def test_stwfsa_train(document_corpus, project, datadir):
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id=StwfsaBackend.name,
        config_params=_backend_conf,
        project=project)
    stwfsa.train(document_corpus)
    assert stwfsa._model is not None
    model_file = datadir.join(stwfsa.MODEL_FILE)
    assert model_file.exists()
    assert model_file.size() > 0


def test_empty_corpus(project):
    corpus = annif.corpus.DocumentList([])
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id=StwfsaBackend.name,
        config_params=dict(),
        project=project)
    with pytest.raises(NotSupportedException):
        stwfsa.train(corpus)


def test_cached_corpus(project):
    corpus = 'cached'
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id=StwfsaBackend.name,
        config_params=dict(),
        project=project)
    with pytest.raises(NotSupportedException):
        stwfsa.train(corpus)


def test_stwfsa_suggest_unknown(project):
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id=StwfsaBackend.name,
        config_params=dict(),
        project=project)
    results = stwfsa.suggest('1234')
    assert len(results) == 0


def test_stwfsa_suggest(project, datadir):
    stwfsa_type = get_backend(StwfsaBackend.name)
    stwfsa = stwfsa_type(
        backend_id=StwfsaBackend.name,
        config_params=dict(),
        project=project)
    # Just some randomly selected words, taken from YSO archaeology group.
    # And "random" words between them
    results = stwfsa.suggest("""random
    muinais-DNA random random
    labyrintit random random random
    Eurooppalainen yleissopimus arkeologisen perinnön suojelusta random
    Indus-kulttuuri random random random random
    kiinteät muinaisjäännökset random random
    makrofossiilit random
    Mesa Verde random random random random
    muinaismuistoalueet  random random random
    zikkuratit random random
    termoluminesenssi random random random""")
    assert len(results) == 10
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p14174' in [
        result.uri for result in hits]
    assert 'labyrintit' in [result.label for result in hits]
