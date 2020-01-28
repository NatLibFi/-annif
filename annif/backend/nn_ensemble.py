"""Neural network based ensemble backend that combines results from multiple
projects."""


from io import BytesIO
import os.path
import numpy as np
from scipy.sparse import csr_matrix
import joblib
import lmdb
from tensorflow.keras.layers import Input, Dense, Add, Flatten, Lambda, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import annif.corpus
import annif.project
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import VectorSuggestionResult
from . import backend
from . import ensemble


class LMDBSequence(Sequence):
    """A sequence of samples stored in a LMDB database."""

    def __init__(self, txn, batch_size):
        self._txn = txn
        cursor = txn.cursor()
        if cursor.last():
            self._counter = int(cursor.key())
        else:  # empty database
            self._counter = 0
        self._batch_size = batch_size

    @staticmethod
    def int_to_key(val):
        return b'%08d' % val

    @staticmethod
    def key_to_int(key):
        return int(key)

    def add_sample(self, inputs, targets):
        # use zero-padded 8-digit key
        key = self.int_to_key(self._counter)
        self._counter += 1
        # convert the sample into a sparse matrix and serialize it as bytes
        sample = (csr_matrix(inputs), csr_matrix(targets))
        buf = BytesIO()
        joblib.dump(sample, buf)
        buf.seek(0)
        self._txn.put(key, buf.read())

    def __getitem__(self, idx):
        """get a particular batch of samples"""
        cursor = self._txn.cursor()
        first_key = idx * self._batch_size
        last_key = first_key + self._batch_size
        cursor.set_key(self.int_to_key(first_key))
        input_arrays = []
        target_arrays = []
        for key, value in cursor.iternext():
            if self.key_to_int(key) >= last_key:
                break
            input_csr, target_csr = joblib.load(BytesIO(value))
            input_arrays.append(input_csr.toarray())
            target_arrays.append(target_csr.toarray().flatten())
        return np.array(input_arrays), np.array(target_arrays)

    def __len__(self):
        """return the number of available batches"""
        return int(np.ceil(self._counter / self._batch_size))


class NNEnsembleBackend(
        backend.AnnifLearningBackend,
        ensemble.EnsembleBackend):
    """Neural network ensemble backend that combines results from multiple
    projects"""

    name = "nn_ensemble"

    MODEL_FILE = "nn-model.h5"
    LMDB_FILE = 'nn-train.mdb'
    LMDB_MAP_SIZE = 1024 * 1024 * 1024

    DEFAULT_PARAMS = {
        'nodes': 100,
        'dropout_rate': 0.2,
        'optimizer': 'adam',
        'epochs': 10,
        'learn-epochs': 1,
    }

    # defaults for uninitialized instances
    _model = None

    def default_params(self):
        params = {}
        params.update(super().default_params())
        params.update(self.DEFAULT_PARAMS)
        return params

    def initialize(self):
        if self._model is not None:
            return  # already initialized
        model_filename = os.path.join(self.datadir, self.MODEL_FILE)
        if not os.path.exists(model_filename):
            raise NotInitializedException(
                'model file {} not found'.format(model_filename),
                backend_id=self.backend_id)
        self.debug('loading Keras model from {}'.format(model_filename))
        self._model = load_model(model_filename)

    def _merge_hits_from_sources(self, hits_from_sources, params):
        score_vector = np.array([hits.vector * weight
                                 for hits, weight in hits_from_sources],
                                dtype=np.float32)
        results = self._model.predict(
            np.expand_dims(score_vector.transpose(), 0))
        return VectorSuggestionResult(results[0], self.project.subjects)

    def _create_model(self, sources):
        self.info("creating NN ensemble model")

        inputs = Input(shape=(len(self.project.subjects), len(sources)))

        flat_input = Flatten()(inputs)
        drop_input = Dropout(
            rate=float(
                self.params['dropout_rate']))(flat_input)
        hidden = Dense(int(self.params['nodes']),
                       activation="relu")(drop_input)
        drop_hidden = Dropout(rate=float(self.params['dropout_rate']))(hidden)
        delta = Dense(len(self.project.subjects),
                      kernel_initializer='zeros',
                      bias_initializer='zeros')(drop_hidden)

        mean = Lambda(lambda x: K.mean(x, axis=2))(inputs)

        predictions = Add()([mean, delta])

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.compile(optimizer=self.params['optimizer'],
                            loss='binary_crossentropy',
                            metrics=['top_k_categorical_accuracy'])

        summary = []
        self._model.summary(print_fn=summary.append)
        self.debug("Created model: \n" + "\n".join(summary))

    def _train(self, corpus, params):
        if corpus == 'cached':
            raise NotSupportedException(
                'Training nn_ensemble project from cached data not supported.')
        sources = annif.util.parse_sources(self.params['sources'])
        self._create_model(sources)
        self._fit_model(corpus, epochs=int(params['epochs']))

    def _corpus_to_vectors(self, corpus, seq):
        # pass corpus through all source projects
        sources = [(annif.project.get_project(project_id), weight)
                   for project_id, weight
                   in annif.util.parse_sources(self.params['sources'])]

        for doc in corpus.documents:
            doc_scores = []
            for source_project, weight in sources:
                hits = source_project.suggest(doc.text)
                doc_scores.append(hits.vector * weight)
            score_vector = np.array(doc_scores,
                                    dtype=np.float32).transpose()
            subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
            true_vector = subjects.as_vector(self.project.subjects)
            seq.add_sample(score_vector, true_vector)

    def _fit_model(self, corpus, epochs):
        lmdb_path = os.path.join(self.datadir, self.LMDB_FILE)
        env = lmdb.open(lmdb_path, map_size=self.LMDB_MAP_SIZE, writemap=True)
        with env.begin(write=True, buffers=True) as txn:
            seq = LMDBSequence(txn, batch_size=32)
            self._corpus_to_vectors(corpus, seq)

            # fit the model
            self._model.fit(seq, verbose=True, epochs=epochs)

        annif.util.atomic_save(
            self._model,
            self.datadir,
            self.MODEL_FILE)

    def _learn(self, corpus, params):
        self.initialize()
        self._fit_model(corpus, int(params['learn-epochs']))
