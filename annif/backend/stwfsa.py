import os
from stwfsapy.predictor import StwfsapyPredictor
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import ListSuggestionResult, SubjectSuggestion
from . import backend
from annif.util import atomic_save, boolean


_KEY_CONCEPT_TYPE_URI = 'concept_type_uri'
_KEY_SUBTHESAURUS_TYPE_URI = 'sub_thesaurus_type_uri'
_KEY_THESAURUS_RELATION_TYPE_URI = 'thesaurus_relation_type_uri'
_KEY_THESAURUS_RELATION_IS_SPECIALISATION = (
    'thesaurus_relation_is_specialisation')
_KEY_REMOVE_DEPRECATED = 'remove_deprecated'
_KEY_HANDLE_TITLE_CASE = 'handle_title_case'
_KEY_EXTRACT_UPPER_CASE_FROM_BRACES = 'extract_upper_case_from_braces'
_KEY_EXTRACT_ANY_CASE_FROM_BRACES = 'extract_any_case_from_braces'
_KEY_EXPAND_AMPERSAND_WITH_SPACES = 'expand_ampersand_with_spaces'
_KEY_EXPAND_ABBREVIATION_WITH_PUNCTUATION = (
    'expand_abbreviation_with_punctuation')
_KEY_SIMPLE_ENGLISH_PLURAL_RULES = 'simple_english_plural_rules'
_KEY_USE_TXT_VEC = 'use_txt_vec'


class StwfsaBackend(backend.AnnifBackend):

    name = "stwfsa"
    needs_subject_index = True

    STWFSA_PARAMETERS = {
        _KEY_CONCEPT_TYPE_URI: str,
        _KEY_SUBTHESAURUS_TYPE_URI: str,
        _KEY_THESAURUS_RELATION_TYPE_URI: str,
        _KEY_THESAURUS_RELATION_IS_SPECIALISATION: boolean,
        _KEY_REMOVE_DEPRECATED: boolean,
        _KEY_HANDLE_TITLE_CASE: boolean,
        _KEY_EXTRACT_UPPER_CASE_FROM_BRACES: boolean,
        _KEY_EXTRACT_ANY_CASE_FROM_BRACES: boolean,
        _KEY_EXPAND_AMPERSAND_WITH_SPACES: boolean,
        _KEY_EXPAND_ABBREVIATION_WITH_PUNCTUATION: boolean,
        _KEY_SIMPLE_ENGLISH_PLURAL_RULES: boolean,
        _KEY_USE_TXT_VEC: bool,
    }

    DEFAULT_PARAMETERS = {
        _KEY_CONCEPT_TYPE_URI: 'http://www.w3.org/2004/02/skos/core#Concept',
        _KEY_SUBTHESAURUS_TYPE_URI:
            'http://www.w3.org/2004/02/skos/core#Collection',
        _KEY_THESAURUS_RELATION_TYPE_URI:
            'http://www.w3.org/2004/02/skos/core#member',
        _KEY_THESAURUS_RELATION_IS_SPECIALISATION: True,
        _KEY_REMOVE_DEPRECATED: True,
        _KEY_HANDLE_TITLE_CASE: True,
        _KEY_EXTRACT_UPPER_CASE_FROM_BRACES: True,
        _KEY_EXTRACT_ANY_CASE_FROM_BRACES: False,
        _KEY_EXPAND_AMPERSAND_WITH_SPACES: True,
        _KEY_EXPAND_ABBREVIATION_WITH_PUNCTUATION: True,
        _KEY_SIMPLE_ENGLISH_PLURAL_RULES: False,
        _KEY_USE_TXT_VEC: False
    }

    MODEL_FILE = 'stwfsa_predictor.zip'

    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug(f'Loading STWFSA model from {path}.')
            if os.path.exists(path):
                self._model = StwfsapyPredictor.load(path)
                self.debug('Loaded model.')
            else:
                raise NotInitializedException(
                    f'Model not found at {path}',
                    backend_id=self.backend_id)

    def _load_data(self, corpus):
        if corpus == 'cached':
            raise NotSupportedException(
                'Training stwfsa project from cached data not supported.')
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train stwfsa project with no documents.')
        self.debug("Transforming training data.")
        X = []
        y = []
        for doc in corpus.documents:
            X.append(doc.text)
            y.append(doc.uris)
        return X, y

    def _train(self, corpus, params):
        X, y = self._load_data(corpus)
        new_params = {
                key: self.STWFSA_PARAMETERS[key](val)
                for key, val
                in params.items()
                if key in self.STWFSA_PARAMETERS
            }
        p = StwfsapyPredictor(
            graph=self.project.vocab.as_graph(),
            langs=frozenset([params['language']]),
            **new_params)
        p.fit(X, y)
        self._model = p
        atomic_save(
            p,
            self.datadir,
            self.MODEL_FILE,
            lambda model, store_path: model.store(store_path))

    def _suggest(self, text, params):
        self.debug(
            f'Suggesting subjects for text "{text[:20]}..." (len={len(text)})')
        result = self._model.suggest_proba([text])[0]
        suggestions = []
        for uri, score in result:
            subject_id = self.project.subjects.by_uri(uri)
            if subject_id:
                label = self.project.subjects[subject_id][1]
            else:
                label = None
            suggestion = SubjectSuggestion(
                uri,
                label,
                None,
                score)
            suggestions.append(suggestion)
        return ListSuggestionResult(suggestions)
