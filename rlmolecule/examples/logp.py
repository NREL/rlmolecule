import os
import sys

from rdkit.Chem import RDConfig
from rdkit.Chem.Descriptors import MolLogP
from rlmolecule.molecule_state import MoleculeState

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


def get_largest_ring_size(molecule):
    cycle_list = molecule.GetRingInfo().AtomRings()
    return max([len(j) for j in cycle_list]) if cycle_list else 0


def penalized_logp(molecule):
    log_p = MolLogP(molecule)
    sa_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sa_score - cycle_score


class PenalizedLogPState(MoleculeState):
    @property
    def reward(self) -> float:
        if self.forced_terminal:
            return penalized_logp(self.molecule)
        else:
            return 0.0
