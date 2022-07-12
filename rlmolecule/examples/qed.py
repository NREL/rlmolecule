import logging
from typing import Optional

import ray
from rdkit.Chem.QED import qed
from rlmolecule.actors import CSVActorWriter
from rlmolecule.molecule_state import MoleculeState

logger = logging.getLogger(__name__)


class QEDState(MoleculeState):
    def __init__(
        self,
        *args,
        use_ray: Optional[bool] = None,
        filename: str = "qed_results.csv",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if use_ray is None:
            use_ray = ray.is_initialized()

        if use_ray:
            self.csv_writer = CSVActorWriter.options(
                name="qed_results", lifetime="detached", get_if_exists=True
            ).remote(filename)
        else:
            self.csv_writer = None

    @property
    def reward(self) -> float:
        if self.forced_terminal:
            reward = qed(self.molecule)
            if self.csv_writer is not None:
                self.csv_writer.write.remote([self.smiles, reward])
            else:
                logger.info(f"QED: {self.smiles} - {reward}")

            return reward
        else:
            return 0.0
