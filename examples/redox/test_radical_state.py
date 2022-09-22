import rdkit
from examples.redox.radical_state import RadicalState
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.molecule_state import MoleculeData

builder = MoleculeBuilder(max_atoms=5)

rstate = RadicalState(
    rdkit.Chem.MolFromSmiles("C"),
    MoleculeData(builder),
)

terminal_state = rstate.children[0].children[4].children[-1]

print(terminal_state)
print(terminal_state.calc_reward())
