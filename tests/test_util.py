"""Unit tests for Annif utility functions"""

import annif.util


def test_boolean():
    inputs = [
        '1',
        '0',
        'true',
        'false',
        'TRUE',
        'FALSE',
        'Yes',
        'No',
        True,
        False]
    outputs = [True, False, True, False, True, False, True, False, True, False]

    for input, output in zip(inputs, outputs):
        assert annif.util.boolean(input) == output


def test_detect_language():
    inputs = (
        'Cat and dog walked on a street.',
        'Kissa ja koira kävelivät kadulla.',
        'En katt och en hund gick ner på gatan.',
        'Cat & koira prom gatan.'  # Nonsense
    )
    outputs = ('en', 'fi', 'sv', None)
    for text, true_lang in zip(inputs, outputs):
        assert true_lang == annif.util.detect_language(text)[0]
