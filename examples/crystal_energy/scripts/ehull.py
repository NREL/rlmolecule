from copy import deepcopy

import numpy as np
import re
from scripts import stability
from scripts import nrelmatdbtaps


# --------------------------------------------------------------------------------------------
# function to alphabetically sort the composition e.g., Li1Sc1F4 would be F4Li1Sc1
# and also get the individual elements in the composition
def sort_comp(comp):
    # split by the digits
    # e.g., for "Li1Sc1F4": ['Li', '1', 'Sc', '1', 'F', '4', '']
    split = np.asarray(re.split('(\d+)', comp))
    ele_stoichs = [''.join(split[i:i+2]) for i in range(0, len(split), 2)]
    sorted_comp = ''.join(sorted(ele_stoichs))
    # also return the elements
    elements = tuple(sorted(split[range(0, len(split) - 1, 2)]))
    return sorted_comp, elements


# --------------------------------------------------------------------------------------------
# function to compute the decomposition energy for a given composition 
def convex_hull_stability(comp, predicted_energy, df_competing_phases):
    """
    :param comp: composition such as Li1Sc1F4
    :param predicted_energy: predicted eV/atom for the structure corresponding to this composition
    :param df_competing_phases: pandas dataframe of competing phases used to 
        construct the convex hull for the elements of the given composition
    """
    comp, eles = sort_comp(comp)

    df_cp = df_competing_phases.copy()
    # UPDATE: try including the composition if it is there
    df_cp = df_cp[df_cp['reduced_composition'] != comp]
    df_cp = df_competing_phases.append({'sortedformula': comp,
                                        'energyperatom': predicted_energy,
                                        'reduced_composition': comp},
                                       ignore_index=True)

    # Create input file for stability analysis
    inputs = nrelmatdbtaps.create_input_DFT(eles, df_cp, chempot='ferev2')
    # if this function failed to create the input, then skip this structure
    if inputs is None:
        return

    # Run stability function (args: input filename, composition)
    try:
        stable_state = stability.run_stability(inputs, comp)
        if stable_state == 'UNSTABLE':
            stoic = frac_stoic(comp)
            hull_nrg = unstable_nrg(stoic, comp, inputs)
            #print("energy above hull of this UNSTABLE phase is", hull_nrg, "eV/atom")
        elif stable_state == 'STABLE':
            stoic = frac_stoic(comp)
            hull_nrg = stable_nrg(stoic, comp, inputs)
            #print("energy above hull of this STABLE phase is", hull_nrg, "eV/atom")
        else:
            print(f"ERR: unrecognized stable_state: '{stable_state}'.")
            print(f"\tcomp: {comp}")
            return
    except SystemError as e:
        print(e)
        print(f"Failed at stability.run_stability for {comp} "
              f"(pred_energy: {predicted_energy}). Skipping\n")
        return
    return hull_nrg


# --------------------------------------------------------------------------------------------
# function to calculate fractional stoichiometry using sortedformula
# e.g., sortedformula = 'Na1Cl1', frac = [0.5,0.5]
def frac_stoic(sortedformula):
    comp_, stoic_ = stability.split_formula(sortedformula)
    stoic_ = list(map(int, stoic_))
    frac = list(np.zeros(len(stoic_)))
    for zz in range(len(frac)):
        frac[zz] = stoic_[zz] / sum(stoic_)
    return frac


# ---------------------------------------------------------------------------------------------------
# function to compute predcited energy above the hull for an UNSTABLE phase (args: fractional stoichiometry and phase)
def unstable_nrg(stoich, phase, inputs):
    stoich_dummy = deepcopy(stoich)

    # A_,B,els_,stoich_ = stability.read_input('input_'+phase)  # read input file for stoichiometry list
    A_, B, els_, stoich_ = stability.read_input(inputs)  # read input file for stoichiometry list

    for j in range(len(A_)):  # loop over all stoichiometries in A_ to find the phase of interest
        if list(A_[j]) == stoich:
            index = j
            # print('phase found')

    original_nrg = B[index]  # save the original formation energy corresponding index found in the above loop

    discrete_nrgs = np.arange(0.001, 6.002, 0.002)  # discretize formation energies
    for jj in range(len(discrete_nrgs)):
        discrete_nrgs[jj] = B[index] - discrete_nrgs[jj]

    ehull = None
    for ii in range(len(discrete_nrgs)):  # for each discretized formation energy, check if STABILITY is achieved
        B[index] = discrete_nrgs[ii]
        if stability.run_stability_hull(inputs, phase, B) == 'STABLE':
            # print("STABILITY CRITERIA ACHIEVED")
            ehull = original_nrg - discrete_nrgs[ii]
            break

    return ehull


# ------------------------------------------------------------------------------------------------------------------
# function to compute decomposition energy for a STABLE phase (args: fractional stoichiometry and phase)
def stable_nrg(stoich, phase, inputs):
    stoich_dummy = deepcopy(stoich)

    A_, B, els_, stoich_ = stability.read_input(inputs)  # read input file for stoichiometry list

    for j in range(len(A_)):  # loop over all stoichiometries in A_ to find the phase of interest
        if list(A_[j]) == stoich:
            index = j
            # print('phase found')

    original_nrg = B[index]  # save the original formation energy corresponding index found in the above loop

    discrete_nrgs = np.arange(0.001, 6.002, 0.002)  # discretize formation energies
    for jj in range(len(discrete_nrgs)):
        discrete_nrgs[jj] = B[index] + discrete_nrgs[jj]

    Hd = None
    for ii in range(len(discrete_nrgs)):  # for each discretized formation energy, check if UNSTABILITY is achieved
        B[index] = discrete_nrgs[ii]
        if stability.run_stability_hull(inputs, phase, B) == 'UNSTABLE':
            # print("UNSTABILITY CRITERIA ACHIEVED")
            Hd = original_nrg - discrete_nrgs[ii]
            break

    return Hd


'''
#--------------------------------------------------------------
# loop over all structures to compute DFT energy above the hull
for i in range(len(df)):
     if df.DFT_stability[i] == 'UNSTABLE':
         stoic = frac_stoic(df.alphabetical_formula[i])
         nrg_hull = e_hull_DFT(stoic,df.alphabetical_formula[i])
         df['ehull_DFT'].values[i] = nrg_hull

     else:
         stoic = frac_stoic(df.alphabetical_formula[i])
         nrg_hull = Hd_DFT(stoic,df.alphabetical_formula[i])
         df['ehull_DFT'].values[i] = nrg_hull
     


# loop over all structures to compute predicted energy above the hull
for i in range(len(df)):
     if df.predicted_stability[i] == 'UNSTABLE':
         stoic = frac_stoic(df.alphabetical_formula[i])
         nrg_hull = e_hull_pred(stoic,df.alphabetical_formula[i])
         df['ehull_predicted'].values[i] = nrg_hull

     else:
         stoic = frac_stoic(df.alphabetical_formula[i])
         nrg_hull = Hd_pred(stoic,df.alphabetical_formula[i])
         df['ehull_predicted'].values[i] = nrg_hull


df.to_parquet('decomposition.parquet',index=False)
'''
