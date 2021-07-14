"""A simple transformation that truncates the text of input documents to a
given character length."""

from annif.exception import ConfigurationException
from . import transformer


class InputLimiter(transformer.IdentityTransform):

    name = 'limit'

    def __init__(self, project, input_limit):
        super().__init__(project)
        self.input_limit = int(input_limit)
        self._validate_value(self.input_limit)

    def transform_fn(self, text):
        return text[:self.input_limit]

    def _validate_value(self, input_limit):
        if input_limit < 0:
            raise ConfigurationException(
                'input_limit in limit_input transformer cannot be negative',
                project_id=self.project.project_id)
