import os

import numpy as np
import pooch
from graphenv import tf
from rlmolecule.policy.preprocessor import load_preprocessor

redox_url = "https://github.com/pstjohn/redox-models/releases/download/v0.0.2/"

redox_model_files = pooch.retrieve(
    redox_url + "redox_model.tar.gz",
    known_hash="sha256:27a3f8db36ca488bf380668660ad1e809de5447e"
    "4d0177c6d996c8cf83935d84",
    processor=pooch.Untar(extract_dir="redox_model"),
)

stability_model_files = pooch.retrieve(
    redox_url + "stability_model.tar.gz",
    known_hash="sha256:78e89eebedc1bcfed4c599db70e92c94b18de9263"
    "131f48ef7b95372ea09254f",
    processor=pooch.Untar(extract_dir="stability_model"),
)

redox_model = tf.keras.models.load_model(
    os.path.dirname(redox_model_files[0]), compile=False
)
stability_model = tf.keras.models.load_model(
    os.path.dirname(stability_model_files[0]), compile=False
)

preprocessor_url = (
    "https://raw.githubusercontent.com/pstjohn/redox-models"
    "/main/models/preprocessor.json"
)
preprocessor_sha = (
    "sha256:264c3ca197b8514aef2b37268366d6542377b6" "a149600e453aeefbc588446304"
)

preprocessor = load_preprocessor(
    pooch.retrieve(
        preprocessor_url,
        known_hash=preprocessor_sha,
    )
)


def expand_inputs(inputs):
    return {key: tf.constant(np.expand_dims(val, 0)) for key, val in inputs.items()}
