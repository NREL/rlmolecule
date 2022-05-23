""" Script to build the action space for decorating crystal structures 
in the RL search. Action space: 

Action space:
1. Choose the elements desired for the battery material 
    i.e., conducting ion, framework cation(s), and anion(s)
2. For a given combination of elements, randomly select a composition 
    from one of the valence-balanced compounds available as a lookup table
3. For the selected composition type, a number of prototype structures are available, 
    which will be classified by their crystal system (cubic, hexagonal, ...). Choose a crystal system randomly.
4. For a chosen crystal system, consider all the prototypes and construct hypothetical decorated structures.
5. Compute the reward for the decorated structure: 
    Predict the total energy using a surrogate model (e.g., GNN), and compute the decomposition energy
"""

import argparse
import os
import sys
import re
import json
import gzip
import itertools
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

#from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core import Composition, Structure


# ### Compositions
# Elements commonly found in battery materials:
# - Conducting ion (C): Li+, Na+, K+, Mg2+, Zn2+
# - Anion (A): F-, Cl-, Br-, I-, O2-, S2-, N3-, P3-
# - Framework cation (F): Sc3+, Y3+, La3+, Ti4+, Zr4+, Hf4+, W6+, Zn2+, Cd2+, Hg2+, B3+, Al3+, Si4+, Ge4+, Sn4+, P5+, Sb5+
# 
# Hypothetical compositions using combinations of C, F, and A are of the following forms:
# 1. Cx Az
# 2. Cx A1z1 A2z2
# 3. Cx Fy Az
# 4. Cx Fy A1z1 A2z2
# 5. Cx F1y1 F2y2 Az
# 6. Cx F1y1 F2y2 A1z1 A2z2 
# 
# The following constraints are employed in generating the compositions:
# 1. A composition may contain only one C ion, up to two (0-2) F ions and at least one and up to two (1-2) A ions.
# 2. The sum of stoichiometric coefficients of the ions is less than or equal to ten, i.e., x + y1 + y2 + z1 + z2 ≤ 10 .
# 3. The generated compositions are valence-balanced, i.e., stoichiometric sum of oxidation states of ions equals to 0.

# ### Build Action space graph
# We are going to build a networkx graph of all the possible actions, split into two parts. Actions 1-2, and actions 3-5.


#def get_element_sets():
# want to maximize the volume around only the conducting ions
conducting_ions = set(['Li', 'Na', 'K', 'Mg', 'Zn'])
# UPDATE 2022-05-19: focus on only these conducting ions
#conducting_ions = set(['Li', 'Na', 'K'])
# P is in the anion and framework_cation sets
anions = set(['F', 'Cl', 'Br', 'I', 'O', 'S', 'N', 'P'])
# TEMP 2022-05-19: remove halides 
#anions = set(['O', 'S', 'N', 'P'])
# Zn is in the conducting_ion and framework_cation sets
framework_cations = set(['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg', 'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'])
# TEMP 2022-05-19: remove halides 
#framework_cations = set(['La', 'Ti', 'Zr', 'Hf', 'W', 'Zn', 'Cd', 'Hg', 'B', 'Al', 'Si', 'Ge', 'Sn', 'P', 'Sb'])
elements = conducting_ions | anions | framework_cations
# sort by the length of the string, so that multiple letter elements come first
elements = sorted(elements, key=len, reverse=True)
print("conducting_ions:", conducting_ions)
print("anions:", anions)
print("framework_cations:", framework_cations)
#    return conducting_ions


# For the reinforcement learning to better distinguish
# between the combination of elements,
# adding an element will be a specific action
# 
# For example:
# 1. Choose a conducting ion
# 2. Choose an anion
# 3. Possibly add a framework cation
# 4. Possibly add another anion
# 5. Possibly add another framework cation
def build_element_combination_actions(G):
    """
    *G*: networkx DiGraph of crystal structure actions
    """    

    # Cx is a conducting ion C present x times 
    # Fy is a framework cation F present y times
    # Az is an anion A present z times
    # below are the 6 different ways to combine the C, F, and Z elements
    for c in conducting_ions:
    # 1. Cx Az
        for a1 in anions - {c}:
            # since the ordering of the elements doesn't matter, 
            # only store the sorted version of the elements
            c_a1 = tuple(sorted(set((c, a1))))
            # add edge from c to (c a1)
            G.add_edge(c, c_a1)
                
    # 2. Cx A1z1 A2z2
            for a2 in anions - set(c_a1):
                c_a1_a2 = tuple(sorted(set((c, a1, a2))))
                G.add_edge(c_a1, c_a1_a2)

    # 4. Cx Fy A1z1 A2z2
                for f1 in framework_cations - set(c_a1_a2):
                    c_f1_a1_a2 = tuple(sorted(set((c, f1, a1, a2))))
                    G.add_edge(c_a1_a2, c_f1_a1_a2)

    # 6. Cx F1y1 F2y2 A1z1 A2z2 
                    for f2 in framework_cations - set(c_f1_a1_a2):
                        c_f1_f2_a1_a2 = tuple(sorted(set((c, f1, f2, a1, a2))))
                        G.add_edge(c_f1_a1_a2, c_f1_f2_a1_a2)

    # 3. Cx Fy Az                
            for f1 in framework_cations - set(c_a1):
                c_f1_a1 = tuple(sorted(set((c, f1, a1))))
                G.add_edge(c_a1, c_f1_a1)

    # 5. Cx F1y1 F2y2 Az
                for f2 in framework_cations - set(c_f1_a1):
                    c_f1_f2_a1 = tuple(sorted(set((c, f1, f2, a1))))
                    G.add_edge(c_f1_a1, c_f1_f2_a1)


def split_comp_to_eles_and_type(comp: str):
    """
    Extract the elements and composition type from a given composition
    e.g., _1_1_4 from Li1Sc1F4
    """
    # this splits by the digits
    # e.g., for "Li1Sc1F4": ['Li', '1', 'Sc', '1', 'F', '4', '']
    comp = comp.replace(' ','')
    split = np.asarray(re.split('(\d+)', comp))
    elements = tuple(sorted(split[range(0, len(split) - 1, 2)]))
    stoich = split[range(1, len(split), 2)]
    # sort the stoichiometry to get the correct order of the comp type
    comp_type = '_' + '_'.join(map(str, sorted(map(int, stoich))))
    return elements, comp_type


def build_eles_to_comp(df_comp, G):
    """ Build the mapping from the element combinations in G 
    to the compositions in df_comp
    """
    compositions = df_comp['composition'].to_list()

    # build dictionary of sorted element tuple to the set of compositions with those elements 
    # e.g., {('F', 'Li', 'Sc'): set(Li1Sc1F4, ...), ...}
    eles_to_comp = defaultdict(set)
    for c in compositions:
        eles, _ = split_comp_to_eles_and_type(c)
        eles_to_comp[tuple(sorted(eles))].add(c)
    print(f"{len(compositions)} compositions map to {len(eles_to_comp)} element tuples")

    # limit to the element combinations present in G
    # the nodes that are element combinations are tuples
    graph_ele_combos = set([n for n in G.nodes() if isinstance(n, tuple)])
    G_eles_to_comp = defaultdict(set)
    for eles in set(eles_to_comp.keys()) & graph_ele_combos:
        G_eles_to_comp[eles] = eles_to_comp[eles]
    print(f"{len(G_eles_to_comp)} / {len(eles_to_comp)} element tuples are combinations in G")

    return G_eles_to_comp


def build_graph_from_eles_to_comps(df_comp, limit_to_comp_types=None):
    """ Build graph from elements to compositions
    :param df_comp: dataframe of valence-balanced compositions to consider
    :param limit_to_comp_types: set of composition types to which the compositions will be limited
        e.g., _1_1_2_4
    """
    G = nx.DiGraph()
    # the first state will just be the string 'root'
    root_node = "root"
    for c in conducting_ions:
        G.add_edge(root_node, c)

    build_element_combination_actions(G)
    print(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    eles_to_comp = build_eles_to_comp(df_comp, G)

    #comp_types = set(df_comp['comp_type'].to_list())
    comp_to_comp_type = dict(zip(df_comp['composition'], df_comp['comp_type']))

    # not all of the generated element sets have a valence-balanced composition,
    # i.e., stoichiometric sum of oxidation states of ions equals 0.
    # Shubham already computed which combinations are valid, so limit to those here
    ele_combo_with_comp = set()
    ele_combo_without_comp = set()
    # the nodes that are element combinations are tuples
    graph_ele_combos = [n for n in G.nodes() if isinstance(n, tuple)]
    for ele_combo in graph_ele_combos:
        if ele_combo in eles_to_comp:
            ele_combo_with_comp.add(ele_combo)
        else:
            ele_combo_without_comp.add(ele_combo)

    print(f"{len(ele_combo_with_comp)} out of {len(graph_ele_combos)} "
          f"element combinations have a corresponding valence-balanced composition")
    # delete the non-valid element combinations from the graph
    G.remove_nodes_from(ele_combo_without_comp)
    print(f"\t{len(G.nodes())} nodes remaining in G")

    # Step 2: For a given combination of elements, randomly select a composition from one of the valence-balanced compounds available as a lookup table
    # As a sanity check, see how many compositions there are for a given combination of elements (top 20):
    print("Top 5 element combinations with the greatest number of compositions:")
    print(sorted(
        {eles: len(comps) for eles, comps in eles_to_comp.items()}.items(),
        key=lambda item: item[1], reverse=True)[:5])

    print("example key and value:")
    print(list(eles_to_comp.items())[0])

    print("Adding the edges from the element combinations to the compositions")
    G_comp_types = set()
    G_comps = set()
    if limit_to_comp_types:
        print(f"Limiting to {len(limit_to_comp_types)} comp_types")
        skipped = set()
    for ele_combo, comps in eles_to_comp.items():
        for comp in comps:
            comp_type = comp_to_comp_type[comp]
            # skip comp_types that don't have a prototype structure
            if limit_to_comp_types and comp_type not in limit_to_comp_types:
                skipped.add(comp)
                continue
            G_comp_types.add(comp_type)
            G_comps.add(comp)

            G.add_edge(ele_combo, comp)

    print(f"\t{len(G_comp_types)} comp types among {len(G_comps)} compositions in G")
    if limit_to_comp_types:
        print(f"\tskipped {len(skipped)} compositions without a prototype")

    print(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges in G')
    return G, G_comp_types, G_comps


# ## Second half of graph structure
# - For the secod half of the search space, start from the composition type, and map out to the decorations.
# - We will be able to use this graph to get to the final state after the composition (first graph) has been decided

# ### Step 3: 
# For the selected composition type, a number of prototype structures are available, which will be classified by their crystal system (e.g., cubic, hexagonal, ...). Select a crystal system.
# 
# 
# The crystal systems are determined by "spacegroup numbers", which represents symmetry of a structure. There are 230 spacegroups (1-230).
# 
# Following is the classification of the 7 crystal systems by spacegroups (sg):
# 1. Triclinic: sg 1-2
# 2. Monoclinic: sg 3-15
# 3. Orthorhombic: sg 16-74
# 4. Tetragonal: 75-142
# 5. Trigonal: sg 143-167
# 6. Hexagonal: 168-194
# 7. Cubic: sg 195-230.
# 
# The spacegroup of a prototype structure is present in the structure id: sg33, sg225 etc.
# For example, spacegroup of prototype structure "POSCAR_sg33_icsd_065132" is 33.
# 
def build_G2_mappings(strc_ids, comp_type_prototypes):
    """
    :param strc_ids: list of structure IDs. 
        Space group should be first in '_' delimited string
        e.g., sg19_icsd_066643
    """
    crystal_systems = {'triclinic': set([1, 2]),
                       'monoclinic': set(range(3, 16)),
                       'orthorhombic': set(range(16, 75)),
                       'tetragonal': set(range(75, 143)),
                       'trigonal': set(range(143, 168)),
                       'hexagonal': set(range(168, 195)),
                       'cubic': set(range(195, 231)),
                       }
    sg_num_to_crystal_sys = {n: crystal_sys
                             for crystal_sys, nums in crystal_systems.items()
                             for n in nums}

    print("head and tail of number of prototypes "
          "available to choose from per composition type")
    print(sorted([len(vals) for vals in comp_type_prototypes.values()], reverse=True)[:20])
    print(sorted([len(vals) for vals in comp_type_prototypes.values()])[:20])

    # map the poscar files to their crystal system
    prototype_to_crystal_sys = {}
    for strc_id in strc_ids:
        #space group should be the first item
        sg_num = strc_id.split('_')[0].replace('sg','')
        system = sg_num_to_crystal_sys[int(sg_num)]
        prototype_to_crystal_sys[strc_id] = system

    crystal_sys_prototypes = defaultdict(set)
    for p, c in prototype_to_crystal_sys.items():
        crystal_sys_prototypes[c].add(p)

    for c, p in crystal_sys_prototypes.items():
        print(c, len(p))

    # how many crystal systems are available for each comp_type?
    comp_type_to_crystal_sys = defaultdict(set)
    for comp_type, prototypes in comp_type_prototypes.items():
        for p in prototypes:
            comp_type_to_crystal_sys[comp_type].add(prototype_to_crystal_sys[p])
    # print(list(comp_type_to_crystal_sys.items())[:3])

    print(f"histogram of number of crystal structures per comp_type "
          f"(out of {len(comp_type_to_crystal_sys)} comp_types):")
    for i in range(1, 8):
        num_matching = len([x for x in comp_type_to_crystal_sys.values() if len(x) == i])
        print(f"{num_matching} comp_types have {i} crystal_systems")

    return prototype_to_crystal_sys


def build_graph_comp_type_to_prototype(comp_type_prototypes,
                                       prototype_to_crystal_sys,
                                       ):
    G2 = nx.DiGraph()

    proto_to_action_node = {}
    # For the selected composition type, a number of prototype structures are available, 
    # which will be classified by their crystal system (e.g., cubic, hexagonal, ...). Select a crystal system.
    # 4. For a chosen crystal system, consider all the prototypes and construct hypothetical decorated structures.
    for comp_type in comp_type_prototypes:
    #     crystal_systems = comp_type_to_crystal_sys[comp_type]
    #     for crystal_sys in crystal_systems:
        for proto in comp_type_prototypes[comp_type]:
            crystal_sys = prototype_to_crystal_sys[proto]
            n1 = comp_type
            n2 = n1 + '|' + crystal_sys
            n3 = n2 + '|' + proto
            G2.add_edge(n1, n2)
            G2.add_edge(n2, n3)
            proto_to_action_node[proto] = n3

            # also add the # decorations.
            # Once we have the composition, we will generate the real decorations. 
            # For now, just use an integer placeholder for the decoration number.
            comp_type_stoic = tuple([int(x) for x in comp_type.split('_')[1:]])
            comp_type_permutations = list(itertools.permutations(comp_type_stoic))
            num_decor = len([permu for permu in comp_type_permutations if permu == comp_type_stoic])
            for i in range(1, num_decor + 1):
                n4 = n3 + '|' + str(i)
                G2.add_edge(n3, n4)

    print(f'{G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges')
    return G2, proto_to_action_node


def get_comp_type(strc):
    """
    Extract the composition type from a given composition
    e.g., _1_1_4 from Li1Sc1F4
    """
    # this splits by the digits
    # e.g., for "Li1Sc1F4": ['Li', '1', 'Sc', '1', 'F', '4', '']
    comp = str(Composition(strc.formula).reduced_composition).replace(' ','')
    split = np.asarray(re.split('(\d+)', comp))
    stoich = split[range(1, len(split), 2)]
    # sort the stoichiometry to get the correct order of the comp type
    comp_type = '_' + '_'.join(map(str, sorted(map(int, stoich))))
    return comp_type, comp


def main(proto_strcs, df_comp, out_pref, write_proto_json=False, enum_decors=True):
    """ Builds two graphs:
    1. Graph for selecting elements to the composition (steps 1-2)
    2. Graph for selecting the prototype and decoration (steps 3-4)
    """
    # get the composition type of the structures
    comp_type_prototypes = defaultdict(set)
    comp_types_skipped = defaultdict(set)
    for strc_id, strc in proto_strcs.items():
        comp_type, comp = get_comp_type(strc)
        # skip some problematic prototypes 
        if '0' in comp_type or comp_type == "_2_2":
            comp_types_skipped[comp_type].add((strc_id, comp)) 
            continue
        comp_type_prototypes[comp_type].add(strc_id)

    print(f"\t{len(comp_type_prototypes)} comp_types")
    print(f"\t{len(comp_types_skipped)} comp_types_skipped: {comp_types_skipped}")

    comp_types = set(list(comp_type_prototypes.keys()))

    G, G_comp_types, G_comps = build_graph_from_eles_to_comps(
        df_comp,
        limit_to_comp_types=comp_types)
    # write this network as the action tree
    out_file = f"{out_pref}eles_to_comps.edgelist.gz"
    print(f"writing G to {out_file}\n")
    nx.write_edgelist(G, out_file, delimiter='\t', data=False)

    # Limit the composition types to those with a composition
    # Don't need to limit the proto_strcs here because they will be limited
    # to those with a composition type
    orig_num_comp_types = len(comp_type_prototypes)
    comp_type_prototypes = {ct: ps for ct, ps in comp_type_prototypes.items() \
                            if ct in G_comp_types}
    print(f"{len(comp_type_prototypes)} / {orig_num_comp_types} prototype composition types "
          f"have a corresponding comp type in G.")
    num_protos = len([p for ps in comp_type_prototypes.values() for p in ps])
    print(f"\t{num_protos} / {len(proto_strcs)} prototypes")

    prototype_to_crystal_sys = build_G2_mappings(
        proto_strcs.keys(),
        comp_type_prototypes)
    G2, proto_to_action_node = build_graph_comp_type_to_prototype(
        comp_type_prototypes,
        prototype_to_crystal_sys)
    # Write this network as the action tree
    out_file = f"{out_pref}comp_type_to_decors.edgelist.gz"
    print(f"writing G2 to {out_file}")
    nx.write_edgelist(G2, out_file, delimiter='\t', data=False)

    # Store a version of the prototypes with the action node as the key
    # e.g., 1_1_1_1_6|cubic|icsd_174512
    # This key is how builder.py is setup to access the prototypes
    if write_proto_json:
        proto_strcs_dict = {a: proto_strcs[p].as_dict() for p, a in proto_to_action_node.items()}
        # first make sure the keys are the action nodes
        out_file = f"{out_pref}prototypes.json.gz"
        print(f"writing {out_file}")
        # https://pymatgen.org/usage.html#side-note-as-dict-from-dict
        with gzip.open(out_file, 'w') as out:
            out.write(json.dumps(proto_strcs_dict).encode())

    # print a table of stats
    stats = {"G # nodes":     G.number_of_nodes(),
             "# comps":       len(G_comps),
             "G2 # nodes":    G2.number_of_nodes(),
             "# comp types":  len(comp_type_prototypes),
             "# prototypes":  len(proto_to_action_node),
    }

    if enum_decors:
        # now enumerate all possible decorations given the compositions and prototype structures
        decorations = enumerate_decorations(G, G2)
        stats["# decorations"] = len(decorations)

    print('\t'.join(stats.keys()))
    print('\t'.join([str(s) for s in stats.values()]))
    
    return G, G2


def enumerate_decorations(G, G2):
    """ enumerate all possible decorations given G and G2
    """
    from rlmolecule.crystal.builder import CrystalBuilder
    from rlmolecule.crystal.crystal_state import CrystalState

    builder = CrystalBuilder(G=G, G2=G2)

    root_state = CrystalState('root')

    n = 16*10**6  # estimated number of decorations
    progress_bar = tqdm(total=n)
    visited = set()
    decorations = list(generate_decoration_ids(builder, root_state, visited, progress_bar))
    return decorations


def generate_decoration_ids(builder, state, visited, progress_bar):
    """ DFS to generate all decorations (state string only) from the ICSD prototype structures
    """
    if str(state) in visited:
        return
    children = state.get_next_actions(builder)
    for c in children:
        yield from generate_decoration_ids(builder, c, visited, progress_bar)
        visited.add(str(c))

    if len(children) == 0:
        progress_bar.update(1)
        yield(str(state))
        
        # generate the decorated structure
        #decorated_structure = generate_decoration(state)


## write a version of the file with just the icsd ID that matches 
## the nomenclature we use for the GNN
## e.g., 'icsd_174512' instead of '_1_1_1_2_3|orthorhombic|POSCAR_sg62_icsd_174512'
#new_structures = {'_'.join(key.split('_')[-2:]): val                   for key, val in new_structures.items()}
#
#out_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/icsd_prototypes.json.gz"
## https://pymatgen.org/usage.html#side-note-as-dict-from-dict
#with gzip.open(out_file, 'w') as out:
#    out.write(json.dumps(
#        {k: s.as_dict() for k, s in new_structures.items()}, 
#        indent=2).encode())


def read_structures_file(structures_file):
    print(f"reading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = json.loads(f.read().decode())
    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)
    print(f"\t{len(structures)} structures read")
    return structures


if __name__ == "__main__":
    # start with either a list of prototype structure files,
    # or a single json file with the prototype structures inside
    parser = argparse.ArgumentParser(
        description='Build the action space for building crystals using '
                    'reinforcement learning through the rlmolecule package')
    parser.add_argument('--prototypes-json', '-j',
                        type=Path,
                        action='append',
                        help="json file(s) containing prototype structures. "
                        "Must specify either this and/or --prototype-files.")
    parser.add_argument('--prototype-files', '-f',
                        type=Path,
                        action='append',
                        help="File(s) containing the paths to the prototype POSCAR files")
    parser.add_argument('--limit-to-ids',
                        type=Path,
                        help="File containing list of prototype IDs to use")
    parser.add_argument('--comp-file',
                        type=Path,
                        default= "../../rlmolecule/crystal/inputs/compositions.csv.gz",
                        help="File containing the compositions to use for decorate")
    parser.add_argument('--out-pref', '-o',
                        type=Path,
                        required=True,
                        help="Output prefix for the two action graphs: "
                        "1) elements to composition, "
                        "2) composition_type to prototype structure")
    parser.add_argument('--write-proto-json',
                        action='store_true',
                        help="Write the prototype structures to a file, "
                        "where the key will be the action node e.g., 1_1_1_1_6|cubic|icsd_174512")
    parser.add_argument('--enumerate-decorations',
                        action='store_true',
                        help="Enumerate all possible decorations")
    
    args = parser.parse_args()

    # read-in the dataframe containing all compositions to decorate
    # TODO build this on the fly using element combinations
    df_comp = pd.read_csv(args.comp_file)
    print(f"{len(df_comp)} compositions")
    print(df_comp.head(2))

    # first, read in the prototype structures
    proto_strcs = {}
    if args.prototypes_json:
        for strc_json in args.prototypes_json:
            strcs = read_structures_file(strc_json)
            proto_strcs.update(strcs)
    if args.prototype_files:
        for prototype_files in args.prototype_files:
            with open(prototype_files, 'r') as f:
                for strc_file in f:
                    strc = Structure.from_file(strc_file, primitive=False)
                    key = os.path.basename(strc_file)
                    proto_strcs[key] = strc

    print(f"{len(proto_strcs)} prototype structures")
    if len(proto_strcs) == 0:
        print("ERROR: need at least 1 prototype structure. Use either --prototypes-json --prototype-files")
        sys.exit()

    # now make sure the keys of the prototype structures are in the right format
    # e.g., sg19_icsd_066643
    correct_key_protos = {}
    for strc_id, strc in proto_strcs.items():
        if '|' in strc_id:
            strc_id = strc_id.split('|')[-1]
        strc_id = strc_id.replace('POSCAR_', '')
        correct_key_protos[strc_id] = strc
    proto_strcs = correct_key_protos
    print(f"{len(proto_strcs)} prototype structures after correcting keys")

    G, G2 = main(proto_strcs,
                 df_comp,
                 args.out_pref,
                 write_proto_json=args.write_proto_json,
                 enum_decors=args.enumerate_decorations,
                 )
