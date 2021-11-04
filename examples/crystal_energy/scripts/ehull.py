from copy import deepcopy

import numpy as np
from scripts import stability


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
