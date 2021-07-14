"""Project management functionality for Annif"""

import enum
import os.path
from shutil import rmtree
import annif
import annif.transformer
import annif.analyzer
import annif.corpus
import annif.suggestion
import annif.backend
import annif.vocab
from annif.datadir import DatadirMixin
from annif.exception import AnnifException, ConfigurationException, \
    NotSupportedException, NotInitializedException

logger = annif.logger


class Access(enum.IntEnum):
    """Enumeration of access levels for projects"""
    private = 1
    hidden = 2
    public = 3


class AnnifProject(DatadirMixin):
    """Class representing the configuration of a single Annif project."""

    # defaults for uninitialized instances
    _transformer = None
    _analyzer = None
    _backend = None
    _vocab = None
    initialized = False

    # default values for configuration settings
    DEFAULT_ACCESS = 'public'

    def __init__(self, project_id, config, datadir, registry):
        DatadirMixin.__init__(self, datadir, 'projects', project_id)
        self.project_id = project_id
        self.name = config.get('name', project_id)
        self.language = config['language']
        self.analyzer_spec = config.get('analyzer', None)
        self.transform_spec = config.get('input_transform', None)
        self.vocab_id = config.get('vocab', None)
        self.config = config
        self._base_datadir = datadir
        self.registry = registry
        self._init_access()

    def _init_access(self):
        access = self.config.get('access', self.DEFAULT_ACCESS)
        try:
            self.access = getattr(Access, access)
        except AttributeError:
            raise ConfigurationException(
                "'{}' is not a valid access setting".format(access),
                project_id=self.project_id)

    def _initialize_analyzer(self):
        if not self.analyzer_spec:
            return  # not configured, so assume it's not needed
        analyzer = self.analyzer
        logger.debug("Project '%s': initialized analyzer: %s",
                     self.project_id,
                     str(analyzer))

    def _initialize_subjects(self):
        try:
            subjects = self.subjects
            logger.debug("Project '%s': initialized subjects: %s",
                         self.project_id,
                         str(subjects))
        except AnnifException as err:
            logger.warning(err.format_message())

    def _initialize_backend(self):
        logger.debug("Project '%s': initializing backend", self.project_id)
        try:
            if not self.backend:
                logger.debug("Cannot initialize backend: does not exist")
                return
            self.backend.initialize()
        except AnnifException as err:
            logger.warning(err.format_message())

    def initialize(self):
        """initialize this project and its backend so that they are ready to
        be used"""

        logger.debug("Initializing project '%s'", self.project_id)

        self._initialize_analyzer()
        self._initialize_subjects()
        self._initialize_backend()

        self.initialized = True

    def _suggest_with_backend(self, text, backend_params):
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        hits = self.backend.suggest(text, beparams)
        logger.debug(
            'Got %d hits from backend %s',
            len(hits), self.backend.backend_id)
        return hits

    @property
    def analyzer(self):
        if self._analyzer is None:
            if self.analyzer_spec:
                self._analyzer = annif.analyzer.get_analyzer(
                    self.analyzer_spec)
            else:
                raise ConfigurationException(
                    "analyzer setting is missing (and needed by the backend)",
                    project_id=self.project_id)
        return self._analyzer

    @property
    def transformer(self):
        if self._transformer is None:
            if self.transform_spec:
                self._transformer = annif.transformer.get_transform(
                    self.transform_spec, project=self)
            else:
                self._transformer = annif.transformer.get_transform(
                    'pass', project=self)
        return self._transformer

    @property
    def backend(self):
        if self._backend is None:
            if 'backend' not in self.config:
                raise ConfigurationException(
                    "backend setting is missing", project_id=self.project_id)
            backend_id = self.config['backend']
            try:
                backend_class = annif.backend.get_backend(backend_id)
                self._backend = backend_class(
                    backend_id, config_params=self.config,
                    project=self)
            except ValueError:
                logger.warning(
                    "Could not create backend %s, "
                    "make sure you've installed optional dependencies",
                    backend_id)
        return self._backend

    @property
    def vocab(self):
        if self._vocab is None:
            if self.vocab_id is None:
                raise ConfigurationException("vocab setting is missing",
                                             project_id=self.project_id)
            self._vocab = annif.vocab.AnnifVocabulary(self.vocab_id,
                                                      self._base_datadir,
                                                      self.language)
        return self._vocab

    @property
    def subjects(self):
        return self.vocab.subjects

    def _get_info(self, key):
        try:
            be = self.backend
            if be is not None:
                return getattr(be, key)
        except AnnifException as err:
            logger.warning(err.format_message())
            return None

    @property
    def is_trained(self):
        return self._get_info('is_trained')

    @property
    def modification_time(self):
        return self._get_info('modification_time')

    def suggest(self, text, backend_params=None):
        """Suggest subjects the given text by passing it to the backend. Returns a
        list of SubjectSuggestion objects ordered by decreasing score."""
        if not self.is_trained:
            if self.is_trained is None:
                logger.warning('Could not get train state information.')
            else:
                raise NotInitializedException('Project is not trained.')
        logger.debug('Suggesting subjects for text "%s..." (len=%d)',
                     text[:20], len(text))
        text = self.transformer.transform_text(text)
        hits = self._suggest_with_backend(text, backend_params)
        logger.debug('%d hits from backend', len(hits))
        return hits

    def train(self, corpus, backend_params=None):
        """train the project using documents from a metadata source"""
        if corpus != 'cached':
            corpus.set_subject_index(self.subjects)
            corpus = self.transformer.transform_corpus(corpus)
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        self.backend.train(corpus, beparams)

    def learn(self, corpus, backend_params=None):
        """further train the project using documents from a metadata source"""
        corpus.set_subject_index(self.subjects)
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        corpus = self.transformer.transform_corpus(corpus)
        if isinstance(
                self.backend,
                annif.backend.backend.AnnifLearningBackend):
            self.backend.learn(corpus, beparams)
        else:
            raise NotSupportedException("Learning not supported by backend",
                                        project_id=self.project_id)

    def hyperopt(self, corpus, trials, jobs, metric, results_file):
        """optimize the hyperparameters of the project using a validation
        corpus against a given metric"""
        if isinstance(
                self.backend,
                annif.backend.hyperopt.AnnifHyperoptBackend):
            optimizer = self.backend.get_hp_optimizer(corpus, metric)
            return optimizer.optimize(trials, jobs, results_file)

        raise NotSupportedException(
            "Hyperparameter optimization not supported "
            "by backend", project_id=self.project_id)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'name': self.name,
                'language': self.language,
                'backend': {'backend_id': self.config.get('backend')},
                'is_trained': self.is_trained,
                'modification_time': self.modification_time
                }

    def remove_model_data(self):
        """remove the data of this project"""
        datadir_path = self._datadir_path
        if os.path.isdir(datadir_path):
            rmtree(datadir_path)
            logger.info('Removed model data for project {}.'
                        .format(self.project_id))
        else:
            logger.warning('No model data to remove for project {}.'
                           .format(self.project_id))
