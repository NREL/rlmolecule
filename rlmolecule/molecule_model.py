from typing import Dict, Optional, Tuple

from graphenv import tf
from graphenv.graph_model import GraphModel
from nfp.preprocessing import MolPreprocessor
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from rlmolecule.policy.model import policy_model


class BaseMoleculeModel(GraphModel):
    def __init__(
        self,
        *args,
        preprocessor: Optional[MolPreprocessor] = None,
        features: int = 64,
        num_messages: int = 3,
        input_dtype: str = "int64",
        max_atoms: Optional[int] = None,
        max_bonds: Optional[int] = None,
        **kwargs,
    ):
        """Create a GNN policy model used to guide the RL search

        Args:
            preprocessor (Optional[MolPreprocessor], optional): A filename or
                initialized preprocessor used to create integer labels for atoms or
                bonds. Defaults to loading a pre-defined json configuration.

            features (int, optional): width of message passing layers. Defaults to 64.
            num_messages (int, optional): depth of message passing layers. Defaults to 3
            input_dtype (str, optional): input layer dtype. Defaults to "int64".
            max_atoms (Optional[int], optional): Atom input size. Defaults to None.
            max_bonds (Optional[int], optional): Bond input size. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.base_model = policy_model(
            preprocessor=preprocessor,
            features=features,
            num_messages=num_messages,
            input_dtype=input_dtype,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
        )

    def forward_vertex(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.base_model(input_dict)


class MoleculeQModel(BaseMoleculeModel, DistributionalQTFModel):
    pass


class MoleculeModel(BaseMoleculeModel, TFModelV2):
    pass
