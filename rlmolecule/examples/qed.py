import logging
from typing import Optional

import ray
from rdkit.Chem.QED import qed
from rlmolecule.actors import CSVActorWriter
from rlmolecule.molecule_state import MoleculeState

logger = logging.getLogger(__name__)


def get_csv_logger(filename):
    return CSVActorWriter.options(
        name="qed_results", lifetime="detached", get_if_exists=True
    ).remote(filename)


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

        self._using_ray = use_ray
        self.filename = filename

        if self._using_ray:
            self.csv_writer = get_csv_logger(filename)
        else:
            self.csv_writer = None

    def new(self, *args, **kwargs):
        return super().new(
            *args, **kwargs, use_ray=self._using_ray, filename=self.filename
        )

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

    def __setstate__(self, d):
        super().__setstate__(d)
        if d["_using_ray"]:
            self.csv_writer = get_csv_logger(d["filename"])

    def __getstate__(self):
        data = super().__getstate__()
        data["csv_writer"] = None
        return data
