"""Transformation filtering out parts of a text that are in a language
different from the language of the project."""

import annif
from . import transformer

logger = annif.logger


class LangFilter(transformer.IdentityTransform):

    name = 'filter_lang'

    def __init__(self, project, threshold=0.99, input_limit=None):
        super().__init__(project)
        self.threshold = float(threshold)
        # TODO: Is input limit is useful also here for performance?
        self.input_limit = int(input_limit) if input_limit is not None \
            else None

    def transform_fn(self, text):
        retained_sentences = []
        char_cnt = 0
        sentences = self.project.analyzer.tokenize_sentences(text)
        logger.debug(f'Number of input chars {len(text)} ' +
                     f'in {len(sentences)} sentences')
        for sent in sentences:
            # TODO: Concat very short sentence to next one for better accuracy?
            detected_lang, probability = annif.util.detect_language(sent)
            # TODO: Retain also those sentences which detection fails on?
            if detected_lang == self.project.language \
                    and probability >= self.threshold:
                retained_sentences.append(sent)
                char_cnt += len(sent)
            if self.input_limit is not None and char_cnt >= self.input_limit:
                break
        text_out = ' '.join(retained_sentences)
        logger.debug(f'Number of retained chars {len(text_out)} ' +
                     f'in {len(retained_sentences)} sentences')
        return text_out
