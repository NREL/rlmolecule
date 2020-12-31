from typing import List, Optional


class MoleculeConfig:
    def __init__(self,
                 max_atoms: int = 10,
                 min_atoms: int = 4,
                 atom_additions: Optional[List] = None,
                 stereoisomers: bool = True,
                 sa_score_threshold: Optional[float] = 3.,
                 tryEmbedding: bool = True) -> None:
        """A configuration class to contain a number of different molecule construction parameters.

        :param max_atoms: Maximum number of heavy atoms
        :param min_atoms: minimum number of heavy atoms
        :param atom_additions: potential atom types to consider. Defaults to ('C', 'H', 'O')
        :param stereoisomers: whether to consider stereoisomers different molecules
        :param sa_score_threshold: If set, don't construct molecules greater than a given sa_score.
        :param tryEmbedding: Try to get a 3D embedding of the molecule, and if this fails, remote it.
        """
        self.max_atoms = max_atoms
        self.stereoisomers = stereoisomers
        self.min_atoms = min_atoms
        self.atom_additions = atom_additions
        self.sa_score_threshold = sa_score_threshold
        self.tryEmbedding = tryEmbedding
