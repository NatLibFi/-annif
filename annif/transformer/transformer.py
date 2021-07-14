"""Common functionality for transforming text of input documents."""

from annif.corpus import TransformingDocumentCorpus
from annif.exception import ConfigurationException


class IdentityTransform():
    """Transform that does not modify text but simply passes it through. This
    class also acts as a base class for other transformations, which need to
    implement and override the transform function."""

    name = 'pass'

    def __init__(self, project):
        self.project = project

    def transform_fn(self, text):
        return text


class TransformChain():
    """Class instantiating and holding the transformation objects performing
    the actual text transformation."""

    def __init__(self, transform_classes, args, project):
        self.project = project
        self.transforms = self._init_transforms(transform_classes, args)

    def _init_transforms(self, transform_classes, args):
        transforms = []
        for trans, (posargs, kwargs) in zip(transform_classes, args):
            try:
                transforms.append(
                    trans(self.project, *posargs, **kwargs))
            except (ValueError, TypeError):
                raise ConfigurationException(
                    f"Invalid arguments to {trans.name} transform: "
                    f"{posargs}, {kwargs})",
                    project_id=self.project.project_id)
        return transforms

    def transform_text(self, text):
        for trans in self.transforms:
            text = trans.transform_fn(text)
        return text

    def transform_corpus(self, corpus):
        return TransformingDocumentCorpus(corpus, self.transform_text)
