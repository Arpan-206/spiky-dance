"""The spiking-network scorer of spiky-dance.

PROVENANCE: `model.py`, `encode.py`, `stdp.py` and `calibrate.py` hold code
from github.com/Arpan-206/mr-spiky (the user's earlier project). Each file has
a header that lists its source file and all changes. Only the feature source
is different: the network, the STDP rule, its hyperparameters, the attribution
and the calibration method are those of Mr. Spiky. `scorer.py` is new.

Not vendored: `src/infer.py` (it has a substitute scoring mode for missing
weights), the path-based whitening loader, and all artefacts of the
code-review model.
"""

from .scorer import ARTEFACT_FILES, Model, load_model

__all__ = ["ARTEFACT_FILES", "Model", "load_model"]
