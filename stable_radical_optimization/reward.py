import logging
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import psycopg2
import nfp
from rdkit import Chem

import alphazero.config as config
import stable_rad_config

logger = logging.getLogger(__name__)

sys.path.append('/projects/rlmolecule/pstjohn/models/20201031_bde/')
from preprocess_inputs import preprocessor as bde_preprocessor
bde_preprocessor.from_json('/projects/rlmolecule/pstjohn/models/20201031_bde/preprocessor.json')

stability_model = tf.keras.models.load_model(
    '/projects/rlmolecule/pstjohn/models/20210214_radical_stability_new_data/',
    compile=False)

redox_model = tf.keras.models.load_model(
    '/projects/rlmolecule/pstjohn/models/20210214_redox_new_data/',
    compile=False)

bde_model = tf.keras.models.load_model(
    '/projects/rlmolecule/pstjohn/models/20210216_bde_new_nfp/',
    compile=False)

@tf.function(experimental_relax_shapes=True)                
def predict(model, inputs):
    return model.predict_step(inputs)

def prepare_for_bde(smiles):

    mol = Chem.MolFromSmiles(smiles)
    radical_index = None
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetNumRadicalElectrons() != 0:
            assert radical_index == None
            is_radical = True
            radical_index = i

            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            atom.SetNumRadicalElectrons(0)
            break

    radical_rank = Chem.CanonicalRankAtoms(mol, includeChirality=True)[radical_index]

    mol_smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(mol_smiles)

    radical_index_reordered = list(Chem.CanonicalRankAtoms(
        mol, includeChirality=True)).index(radical_rank)

    molH = Chem.AddHs(mol)
    for bond in molH.GetAtomWithIdx(radical_index_reordered).GetBonds():
        if 'H' in {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}:
            bond_index = bond.GetIdx()
            break
        
    h_bond_indices = [bond.GetIdx() for bond in filter(
        lambda bond: ((bond.GetEndAtom().GetSymbol() == 'H') 
                      | (bond.GetBeginAtom().GetSymbol() == 'H')), molH.GetBonds())]
    
    other_h_bonds = list(set(h_bond_indices) - {bond_index})
            
    return pd.Series({
        'mol_smiles': mol_smiles,
        'radical_index_mol': radical_index_reordered,
        'bond_index': bond_index,
        'other_h_bonds': other_h_bonds
    })


def bde_get_inputs(smiles):
    inputs = bde_preprocessor.construct_feature_matrices(smiles, train=False)
    assert not (inputs['atom'] == 1).any() | (inputs['bond'] == 1).any()
    return {key: np.expand_dims(val, 0) for key, val in inputs.items()}


def calc_bde(node):
    """calculate the X-H bde, and the difference to the next-weakest X-H bde in kcal/mol"""
    
    bde_inputs = prepare_for_bde(node.smiles)
    inputs = bde_get_inputs(bde_inputs.mol_smiles)
    
    pred_bdes = predict(bde_model, inputs)[0][0, :, 0].numpy()
    
    bde_radical = pred_bdes[bde_inputs.bond_index]
    
    if len(bde_inputs.other_h_bonds) == 0:
        bde_diff = 30.  # Just an arbitrary large number
    
    else:
        other_h_bdes = pred_bdes[bde_inputs.other_h_bonds]
        bde_diff = (other_h_bdes - bde_radical).min()
    
    return bde_radical, bde_diff


def windowed_loss(target, desired_range):
    """ Returns 0 if the molecule is in the middle of the desired range,
    scaled loss otherwise. """
    
    span = desired_range[1] - desired_range[0]
    
    lower_lim = desired_range[0] + span / 6
    upper_lim = desired_range[1] - span / 6
    
    if target < lower_lim:
        return max(1 - 3*(abs(target - lower_lim) / span), 0)
    elif target > upper_lim:
        return max(1 - 3*(abs(target - upper_lim) / span), 0)
    else:
        return 1


def calc_reward(node):
    
    with psycopg2.connect(**config.dbparams) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select real_reward from {table}_reward where smiles = %s".format(
                    table=config.sql_basename), (node.smiles,))
            result = cur.fetchone()

    if result:
        # Probably would put RR code here?
        rr = get_ranked_rewards(result[0])
        node._true_reward = result[0]
        return rr

    # Node is outside the domain of validity
    elif ((node.policy_inputs['atom'] == 1).any() | 
          (node.policy_inputs['bond'] == 1).any()):
        node._true_reward = 0.
        return config.min_reward

    else:
        reward = calc_reward_inner(node)
        rr = get_ranked_rewards(reward)
        node._true_reward = reward
        return rr        

def calc_reward_inner(node):

    logger.debug(f"Calculating reward for {node}")
    
    spins, buried_vol = predict(stability_model,
        {key: tf.constant(np.expand_dims(val, 0))
         for key, val in node.policy_inputs.items()})

    spins = spins.numpy().flatten()
    buried_vol = buried_vol.numpy().flatten()

    atom_index = int(spins.argmax())
    max_spin = spins[atom_index]
    spin_buried_vol = buried_vol[atom_index]

    atom_type = node.GetAtomWithIdx(atom_index).GetSymbol()

    ionization_energy, electron_affinity = predict(
        redox_model, {key: tf.constant(np.expand_dims(val, 0))
                      for key, val in node.policy_inputs.items()}).numpy().tolist()[0]

    v_diff = ionization_energy - electron_affinity
    bde, bde_diff = calc_bde(node)

    ea_range = (-.5, 0.2)
    ie_range = (.5, 1.2)
    v_range = (1, 1.7)   
    bde_range = (60, 80)    
    

    # This is a bit of a placeholder; but the range for spin is about 1/50th that
    # of buried volume.
    reward = ((1 - max_spin) * 50 + spin_buried_vol 
              + 25 / (1 + np.exp(-(bde_diff - 10)))
              + 100 * (
        windowed_loss(electron_affinity, ea_range) +     
        windowed_loss(ionization_energy, ie_range) + 
        windowed_loss(v_diff, v_range) +         
        windowed_loss(bde, bde_range)) / 4)

       
    with psycopg2.connect(**config.dbparams) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO {table}_reward
                (smiles, real_reward, atom_type, buried_vol, max_spin, atom_index, ie, ea, bde) 
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;""".format(table=config.sql_basename), (
                    node.smiles, float(reward), atom_type, # This should be the real reward
                    float(spin_buried_vol), float(max_spin), atom_index,
                    float(ionization_energy), float(electron_affinity), float(bde)))

    return reward 


def get_ranked_rewards(reward):

    with psycopg2.connect(**config.dbparams) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from {table}_game where experiment_id = %s;".format(
                table=config.sql_basename), (config.experiment_id,))
            n_games = cur.fetchone()[0]
        
        if n_games < config.reward_buffer_min_size:
            # Here, we don't have enough of a game buffer
            # to decide if the move is good or not
            logging.debug(f"ranked_reward: not enough games ({n_games})")
            return np.random.choice([0., 1.])
        
        else:
            with conn.cursor() as cur:
                cur.execute("""
                        select percentile_disc(%s) within group (order by real_reward) 
                        from (select real_reward from {table}_game where experiment_id = %s
                              order by id desc limit %s) as finals
                        """.format(table=config.sql_basename),
                            (config.ranked_reward_alpha, config.experiment_id, config.reward_buffer_max_size))
                
                r_alpha = cur.fetchone()[0]

            logging.debug(f"ranked_reward: r_alpha={r_alpha}, reward={reward}")
            
            if np.isclose(reward, r_alpha):
                return np.random.choice([0., 1.])
            
            elif reward > r_alpha:
                return 1.
            
            elif reward < r_alpha:
                return 0.