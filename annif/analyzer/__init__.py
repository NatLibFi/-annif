"""Collection of language-specific analyzers and analyzer registry for Annif"""

import re
from . import simple
from . import snowball
import annif
from annif.util import parse_args

_analyzers = {}


def register_analyzer(analyzer):
    _analyzers[analyzer.name] = analyzer


def _extend_posargs(posargs):
    if not posargs:
        posargs = [None]
    return posargs


def get_analyzer(analyzerspec):
    match = re.match(r'(\w+)(\((.*)\))?', analyzerspec)
    if match is None:
        raise ValueError(
            "Invalid analyzer specification {}".format(analyzerspec))

    analyzer = match.group(1)
    posargs, kwargs = parse_args(match.group(3))
    posargs = _extend_posargs(posargs)
    try:
        return _analyzers[analyzer](*posargs, **kwargs)
    except KeyError:
        raise ValueError("No such analyzer {}".format(analyzer))


register_analyzer(simple.SimpleAnalyzer)
register_analyzer(snowball.SnowballAnalyzer)

# Optional analyzers
try:
    from . import voikko
    register_analyzer(voikko.VoikkoAnalyzer)
except ImportError:
    annif.logger.debug("voikko not available, not enabling voikko analyzer")
