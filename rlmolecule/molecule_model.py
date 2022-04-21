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
        num_heads: int = 4,
        num_messages: int = 3,
        input_dtype: str = "int64",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_model = policy_model(
            preprocessor, features, num_heads, num_messages, input_dtype
        )

    def forward_vertex(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.base_model(input_dict)


class MoleculeQModel(BaseMoleculeModel, DistributionalQTFModel):
    pass


class MoleculeModel(BaseMoleculeModel, TFModelV2):
    pass
