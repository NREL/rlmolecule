import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from examples.crystal_volume import optimize_crystal_volume as ocv
from examples.crystal_volume.optimize_crystal_volume import CrystalVolOptimizationProblem
from rlmolecule.crystal.builder import CrystalBuilder
from rlmolecule.crystal.crystal_state import CrystalState
from rlmolecule.crystal.crystal_problem import CrystalTFAlphaZeroProblem
from rlmolecule.crystal.preprocessor import CrystalPreprocessor
from rlmolecule.sql.run_config import RunConfig
from rlmolecule.sql import Base, Session


def policy_model_sequential(features: int = 64,
                            num_eles_and_stoich: int = 252,
                            num_crystal_sys: int = 7,
                            num_proto_strc: int = 4170,
                            ) -> tf.keras.Model:
    #crystal_sys_class = layers.Input(shape=[1], dtype=tf.int64, name='crystal_sys')
    #proto_strc_class = layers.Input(shape=[1], dtype=tf.int64, name='proto_strc')
    crystal_sys_model = tf.keras.models.Sequential()
    crystal_sys_model.add(layers.Embedding(num_crystal_sys + 1), features, input_length=1)
    crystal_sys_model.add(layers.Dense(4, activation='relu'))

    # input_tensors = [element_class, crystal_sys_class, proto_strc_class]
    input_tensors = [crystal_sys_class, proto_strc_class]

    # element_embedding = layers.Embedding(
    #    num_eles_and_stoich, features, name='conducting_embedding')(element_class)
    #
    #    elements_output = layers.Dense(features, activation='relu')(element_embedding)

    # TODO don't need an embedding because the number of crystal systems is small(?). Just use a one-hot encoding
    crystal_sys_embedding = layers.Embedding(
        num_crystal_sys + 1, features, name='crystal_sys_embedding')(crystal_sys_class)
    crystal_sys_output = layers.Dense(features, activation='relu')(crystal_sys_embedding)
    crystal_sys_model = tf.keras.Model(crystal_sys_class, outputs=crystal_sys_output)

    proto_strc_embedding = layers.Embedding(
        num_proto_strc + 1, features, name='proto_strc_embedding')(proto_strc_class)
    proto_strc_output = layers.Dense(features, activation='relu')(proto_strc_embedding)
    proto_strc_model = tf.keras.Model(proto_strc_class, outputs=proto_strc_output)

    # Merge all available features into a single large vector via concatenation
    # x = layers.concatenate([elements_output, crystal_sys_output, proto_strc_output])
    x = layers.concatenate([crystal_sys_model.output, proto_strc_model.output])
    global_state = layers.Dense(features, activation='relu')(x)
    output = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, output, name='policy_model')


def policy_model(features: int = 64,
                 num_eles_and_stoich: int = 252,
                 num_crystal_sys: int = 7,
                 num_proto_strc: int = 4170,
                 ) -> tf.keras.Model:
    """ Constructs a policy model that predicts value, pi_logits from a batch of molecule inputs. Main model used in
    policy training and loading weights

    :param preprocessor: a MolPreprocessor class for initializing the embedding matrices
    :param features: Size of network hidden layers
    :return: The constructed policy model
    """
    # Define inputs
    # 5 conducting ions, 8 anions, 17 framework cations, up to 8 elements in a composition.
    # conducting_ion_class = layers.Input(shape=[None], dtype=tf.int65, name='conducting_ion')
    # anion_class = layers.Input(shape=[None], dtype=tf.int65, name='anion')
    # framework_cation_class = layers.Input(shape=[None], dtype=tf.int65, name='framework_cation')
    # I will include the elements by themselves, and the elements with a stoichiometry e.g., 'Cl', 'Cl6'
    # TODO Many element stoichiometries are not present. For now I will just include all of them
    element_class = layers.Input(shape=[10], dtype=tf.int64, name='eles_and_stoich')
    # 7 crystal systems
    crystal_sys_class = layers.Input(shape=[], dtype=tf.int64, name='crystal_sys')
    # 4170 total prototype structures
    proto_strc_class = layers.Input(shape=[], dtype=tf.int64, name='proto_strc')

    input_tensors = [element_class, crystal_sys_class, proto_strc_class]

    element_embedding = layers.Embedding(
        input_dim=num_eles_and_stoich, output_dim=features,
        input_length=None, name='conducting_embedding')(element_class)
    print(element_embedding.shape)
    element_embedding = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=-2, keepdims=True), output_shape=lambda s: (s[-1],))(element_embedding)
    print(element_embedding.shape)
    element_embedding = layers.Reshape((features,))(element_embedding)
    print(element_embedding.shape)
    # embedding_dense = layers.Dense(features, activation='relu')(element_embedding)
    # print(embedding_dense.shape)

    crystal_sys_embedding = layers.Embedding(
        input_dim=num_crystal_sys+1, output_dim=features,
        input_length=1, mask_zero=True, name='crystal_sys_embedding')(crystal_sys_class)
    print(crystal_sys_embedding.shape)
    proto_strc_embedding = layers.Embedding(
        input_dim=num_proto_strc+1, output_dim=features,
        input_length=1, mask_zero=True, name='proto_strc_embedding')(proto_strc_class)
    print(proto_strc_embedding.shape)

    x = layers.concatenate([element_embedding, crystal_sys_embedding, proto_strc_embedding])
    #x = np.sum()

    # crystal_proto = layers.concatenate([crystal_sys_embedding, proto_strc_embedding])
    # crystal_proto_dense = layers.Dense(features, activation='relu')(crystal_proto)
    # max_pool = layers.GlobalMaxPooling1D()(crystal_proto_dense)
    # x = layers.concatenate(element_embedding + [max_pool])

    #    elements_output = layers.Dense(features, activation='relu')(element_embedding)
    # crystal_sys_output = layers.Dense(features, activation='relu')(crystal_sys_embedding)
    # proto_strc_output = layers.Dense(features, activation='relu')(proto_strc_embedding)

    # Merge all available features into a single large vector via concatenation
    #x = layers.concatenate([elements_output, crystal_sys_output, proto_strc_output])
    # x = layers.concatenate([crystal_sys_model.output, proto_strc_model.output])
    global_state = layers.Dense(features, activation='relu')(x)
    output = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, output, name='policy_model')


def test_policy_model(features: int = 64,
                      num_eles_and_stoich: int = 252,
                      num_crystal_sys: int = 7,
                      num_proto_strc: int = 4170,
                 ) -> tf.keras.Model:
    """ Constructs a policy model that predicts value, pi_logits from a batch of molecule inputs. Main model used in
    policy training and loading weights

    :param preprocessor: a MolPreprocessor class for initializing the embedding matrices
    :param features: Size of network hidden layers
    :return: The constructed policy model
    """
    # Define inputs
    # 5 conducting ions, 8 anions, 17 framework cations, up to 8 elements in a composition.
    # I will include the elements by themselves, and the elements with a stoichiometry e.g., 'Cl', 'Cl6'
    # TODO Many element stoichiometries are not present. For now I will just include all of them
    element_class = layers.Input(shape=[num_eles_and_stoich], dtype=tf.int64, name='eles_and_stoich')
    # 7 crystal systems
    crystal_sys_class = layers.Input(shape=[num_crystal_sys], dtype=tf.int64, name='crystal_sys')
    # 4170 total prototype structures
    proto_strc_class = layers.Input(shape=[num_proto_strc], dtype=tf.int64, name='proto_strc')

    input_tensors = [element_class, crystal_sys_class, proto_strc_class]

    # element_embedding = layers.Embedding(
    #     num_eles_and_stoich, features, name='conducting_embedding')(element_class)

    # elements_output = layers.Dense(features, activation='relu')(element_embedding)
    elements_output = layers.Dense(features // 3, activation='relu')(element_class)

    ## TODO don't need an embedding because the number of crystal systems is small(?). Just use a one-hot encoding
    #crystal_sys_embedding = layers.Embedding(
    #    num_crystal_sys, features, name='crystal_sys_embedding')(crystal_sys_class)
    # crystal_sys_output = layers.Dense(features, activation='relu')(crystal_sys_embedding)
    crystal_sys_output = layers.Dense(features // 3, activation='relu')(crystal_sys_class)

    # proto_strc_embedding = layers.Embedding(
    #     num_proto_strc, features, name='proto_strc_embedding')(proto_strc_class)
    # proto_strc_output = layers.Dense(features, activation='relu')(proto_strc_embedding)
    proto_strc_output = layers.Dense(features // 3, activation='relu')(proto_strc_class)

    # Merge all available features into a single large vector via concatenation
    x = layers.Concatenate(axis=0)([elements_output, crystal_sys_output, proto_strc_output])
    global_state = layers.Dense(features, activation='relu')(x)
    output = layers.Dense(1)(global_state)

    return tf.keras.Model(input_tensors, output, name='policy_model')


run_config = RunConfig(None)
ocv.run_config = run_config

engine = run_config.start_engine()
Base.metadata.create_all(engine, checkfirst=True)
Session.configure(bind=engine)
session = Session()
ocv.engine = engine

# now try passing each of these through the policy model to see if they work
# problem = CrystalVolOptimizationProblem(engine)
problem = ocv.create_problem()
# model = policy_model()
model = problem.policy_model()

print(model.summary())

preprocessor = CrystalPreprocessor()
# this state will have elements, composition, crystal system, and structure

#print(model.predict(policy_inputs))

#problem = ocv.create_problem()
#builder = problem.builder
builder = CrystalBuilder()

root = 'root'
state = CrystalState(root)

#print(state.get_next_actions())

state = CrystalState('Li')
print(state)
policy_inputs = preprocessor.construct_feature_matrices(state)
print(policy_inputs)
print(model(policy_inputs))

while state.terminal is False:
    next_actions = state.get_next_actions(builder)
    state = next_actions[-1]
    print(state)

    policy_inputs = preprocessor.construct_feature_matrices(state)
    print(policy_inputs)
    print(model(policy_inputs))

