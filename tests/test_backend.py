"""Unit tests for backends in Annif"""

import pytest
import annif
import annif.backend
import annif.corpus


def test_get_backend_nonexistent():
    with pytest.raises(ValueError):
        annif.backend.get_backend("nonexistent")


def test_get_backend_dummy(project):
    dummy_type = annif.backend.get_backend("dummy")
    dummy = dummy_type(backend_id='dummy', config_params={},
                       project=project)
    result = dummy.suggest(text='this is some text')
    assert len(result) == 1
    hits = result.as_list(project.subjects)
    assert hits[0].uri == 'http://example.org/dummy'
    assert hits[0].label == 'dummy'
    assert hits[0].score == 1.0


def test_learn_dummy(project, tmpdir):
    dummy_type = annif.backend.get_backend("dummy")
    dummy = dummy_type(backend_id='dummy', config_params={},
                       project=project)

    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.tsv').write('<http://example.org/key1>\tkey1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.tsv').write('<http://example.org/key2>\tkey2')
    docdir = annif.corpus.DocumentDirectory(str(tmpdir))

    dummy.learn(docdir)

    result = dummy.suggest(text='this is some text')
    assert len(result) == 1
    hits = result.as_list(project.subjects)
    assert hits[0].uri == 'http://example.org/key1'
    assert hits[0].label == 'key1'
    assert hits[0].score == 1.0


def test_fill_params_with_defaults(project):
    dummy_type = annif.backend.get_backend('dummy')
    dummy = dummy_type(backend_id='dummy', config_params={},
                       project=project)
    expected_default_params = {'limit': 100}
    assert expected_default_params == dummy.params
