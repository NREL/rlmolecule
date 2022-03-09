from typing import Dict, Optional, Tuple

from graphenv import tf
from graphenv.graph_model import GraphModel
from nfp.preprocessing import MolPreprocessor

from rlmolecule.policy.model import policy_model


class MoleculeModel(GraphModel):
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
