"""Common functionality for backends."""

import abc
import os.path
from datetime import datetime, timezone
from glob import glob
from annif import logger


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None
    needs_subject_index = False

    DEFAULT_PARAMETERS = {'limit': 100}

    def __init__(self, backend_id, config_params, project):
        """Initialize backend with specific parameters. The
        parameters are a dict. Keys and values depend on the specific
        backend type."""
        self.backend_id = backend_id
        self.config_params = config_params
        self.project = project
        self.datadir = project.datadir

    def default_params(self):
        return self.DEFAULT_PARAMETERS

    @property
    def params(self):
        params = {}
        params.update(self.default_params())
        params.update(self.config_params)
        return params

    @property
    def is_trained(self):
        return bool(glob(os.path.join(self.datadir, '*')))

    @property
    def modification_time(self):
        mtimes = [datetime.utcfromtimestamp(os.path.getmtime(p))
                  for p in glob(os.path.join(self.datadir, '*'))]
        most_recent = max(mtimes, default=None)
        if most_recent is None:
            return None
        return most_recent.replace(tzinfo=timezone.utc)

    def _get_backend_params(self, params):
        backend_params = dict(self.params)
        if params is not None:
            backend_params.update(params)
        return backend_params

    def _train(self, corpus, params):
        """This method can be overridden by backends. It implements
        the train functionality, with pre-processed parameters."""
        pass  # default is to do nothing, subclasses may override

    def train(self, corpus, params=None):
        """Train the model on the given document or subject corpus."""
        beparams = self._get_backend_params(params)
        return self._train(corpus, params=beparams)

    def initialize(self):
        """This method can be overridden by backends. It should cause the
        backend to pre-load all data it needs during operation."""
        pass

    @abc.abstractmethod
    def _suggest(self, text, params):
        """This method should implemented by backends. It implements
        the suggest functionality, with pre-processed parameters."""
        pass  # pragma: no cover

    def suggest(self, text, params=None):
        """Suggest subjects for the input text and return a list of subjects
        represented as a list of SubjectSuggestion objects."""
        beparams = self._get_backend_params(params)
        self.initialize()
        return self._suggest(text, params=beparams)

    def debug(self, message):
        """Log a debug message from this backend"""
        logger.debug("Backend {}: {}".format(self.backend_id, message))

    def info(self, message):
        """Log an info message from this backend"""
        logger.info("Backend {}: {}".format(self.backend_id, message))

    def warning(self, message):
        """Log a warning message from this backend"""
        logger.warning("Backend {}: {}".format(self.backend_id, message))


class AnnifLearningBackend(AnnifBackend):
    """Base class for Annif backends that can perform online learning"""

    @abc.abstractmethod
    def _learn(self, corpus, params):
        """This method should implemented by backends. It implements the learn
        functionality, with pre-processed parameters."""
        pass  # pragma: no cover

    def learn(self, corpus, params=None):
        """Further train the model on the given document or subject corpus."""
        beparams = self._get_backend_params(params)
        return self._learn(corpus, params=beparams)
