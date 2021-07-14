"""Clases for supporting document corpora"""

import glob
import os.path
import re
import gzip
import annif.util
from itertools import islice
from .types import DocumentCorpus
from .subject import SubjectSet

logger = annif.logger


class DocumentDirectory(DocumentCorpus):
    """A directory of files as a full text document corpus"""

    def __init__(self, path, require_subjects=False):
        self.path = path
        self.require_subjects = require_subjects

    def __iter__(self):
        """Iterate through the directory, yielding tuples of (docfile,
        subjectfile) containing file paths. If there is no key file and
        require_subjects is False, the subjectfile will be returned as None."""

        for filename in sorted(glob.glob(os.path.join(self.path, '*.txt'))):
            tsvfilename = re.sub(r'\.txt$', '.tsv', filename)
            if os.path.exists(tsvfilename):
                yield (filename, tsvfilename)
                continue
            keyfilename = re.sub(r'\.txt$', '.key', filename)
            if os.path.exists(keyfilename):
                yield (filename, keyfilename)
                continue
            if not self.require_subjects:
                yield (filename, None)

    @property
    def documents(self):
        for docfilename, keyfilename in self:
            with open(docfilename, errors='replace',
                      encoding='utf-8-sig') as docfile:
                text = docfile.read()
            with open(keyfilename, encoding='utf-8-sig') as keyfile:
                subjects = SubjectSet.from_string(keyfile.read())
            yield self._create_document(text=text,
                                        uris=subjects.subject_uris,
                                        labels=subjects.subject_labels)


class DocumentFile(DocumentCorpus):
    """A TSV file as a corpus of documents with subjects"""

    def __init__(self, path):
        self.path = path

    @property
    def documents(self):
        if self.path.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open
        with opener(self.path, mode='rt', encoding='utf-8-sig') as tsvfile:
            for line in tsvfile:
                yield from self._parse_tsv_line(line)

    def _parse_tsv_line(self, line):
        if '\t' in line:
            text, uris = line.split('\t', maxsplit=1)
            subjects = [annif.util.cleanup_uri(uri)
                        for uri in uris.split()]
            yield self._create_document(text=text,
                                        uris=subjects,
                                        labels=[])
        else:
            logger.warning('Skipping invalid line (missing tab): "%s"',
                           line.rstrip())


class DocumentList(DocumentCorpus):
    """A document corpus based on a list of other iterable of Document
    objects"""

    def __init__(self, documents):
        self._documents = documents

    @property
    def documents(self):
        yield from self._documents


class TransformingDocumentCorpus(DocumentCorpus):
    """A document corpus that wraps another document corpus but transforms the
    documents using a given transform function"""

    def __init__(self, corpus, transform_fn):
        self._orig_corpus = corpus
        self._transform_fn = transform_fn

    @property
    def documents(self):
        for doc in self._orig_corpus.documents:
            yield self._create_document(
                text=self._transform_fn(doc.text),
                uris=doc.uris,
                labels=doc.labels)


class LimitingDocumentCorpus(DocumentCorpus):
    """A document corpus that wraps another document corpus but limits the
    number of documents to a given limit"""

    def __init__(self, corpus, docs_limit):
        self._orig_corpus = corpus
        self.docs_limit = docs_limit

    @property
    def documents(self):
        for doc in islice(self._orig_corpus.documents, self.docs_limit):
            yield self._create_document(text=doc.text,
                                        uris=doc.uris,
                                        labels=doc.labels)
