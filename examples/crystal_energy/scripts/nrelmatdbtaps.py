from operator import itemgetter

import pandas

# ---FERE reference chemical potentials------
fere_chempot = {'Ag': -0.83, 'Al': -3.02, 'As': -5.06, 'Au': -2.23, 'Ba': -1.39, 'Be': -3.40, 'Bi': -4.39, 'Ca': -1.64,
                'Cd': -0.56, 'Cl': -1.63, 'Co': -4.75, 'Cr': -7.22, 'Cu': -1.97, 'F': -1.70, 'Fe': -6.15, 'Ga': -2.37,
                'Ge': -4.14, 'Hf': -7.40, 'Hg': -0.12, 'In': -2.31, 'Ir': -5.96, 'K': -0.80, 'La': -3.66, 'Li': -1.65,
                'Mg': -0.99, 'Mn': -7.00, 'N': -8.51, 'Na': -1.06, 'Nb': -6.69, 'Ni': -3.57, 'O': -4.76, 'P': -5.64,
                'Pd': -3.12, 'Pt': -3.95, 'Rb': -0.68, 'Rh': -4.76, 'S': -4.00, 'Sb': -4.29, 'Sc': -4.63, 'Se': -3.55,
                'Si': -4.99, 'Sn': -3.79, 'Sr': -1.17, 'Ta': -8.82, 'Te': -3.25, 'Ti': -5.52, 'V': -6.42, 'Y': -4.81,
                'Zn': -0.84, 'Zr': -5.87}

ferev2_chempot = {'Ag': -0.79, 'Al': -3.27, 'Al_anion': -3.55, 'As': -4.95, 'As_cation': -4.42, 'Au': -1.96, 'B': -6.73,
                  'B_anion': -6.44, 'Ba': -1.44, 'Be': -3.50, 'Bi_cation': -4.22, 'Bi': -4.19, 'Br': -1.54, 'C': -8.94,
                  'Ca': -1.78, 'Cd': -0.64, 'Cl': -1.74, 'Co': -4.67, 'Cr': -7.08, 'Cu': -1.87, 'F': -1.44, 'Fe': -6.00,
                  'Ga': -2.53, 'Ge': -4.34, 'Ge_anion': -4.84, 'Hf': -7.38, 'Hg': -0.10, 'I': -1.53, 'In': -2.39,
                  'Ir': -6.31, 'K': -0.79, 'La': -3.76, 'Li': -1.58, 'Mg': -1.23, 'Mo': -7.37, 'Mn': -6.86, 'N': -8.46,
                  'Na': -1.02, 'Nb': -6.92, 'Ni': -3.65, 'O': -4.80, 'P': -5.17, 'P_cation': -5.14, 'Pb': -3.85,
                  'Pb_anion': -4.29, 'Pd': -3.00, 'Pt': -3.88, 'Rb': -0.58, 'Rh': -4.66, 'Ru': -6.14, 'S': -4.01,
                  'Sb_cation': -4.13, 'Sb': -4.16, 'Sc': -4.42, 'Se': -3.54, 'Si': -5.30, 'Si_anion': -5.40,
                  'Sn': -3.87, 'Sn_anion': -3.71, 'Sr': -1.32, 'Ta': -8.82, 'Te': -3.18, 'Te_cation': -2.75,
                  'Ti': -5.39, 'V': -6.40, 'W': -9.61, 'Y': -4.96, 'Zn': -0.94, 'Zr': -6.39}


# ------------------------------------
def split_formula(compound):
    compoundnew = []
    compoundnew2 = []

    for i in range(len(compound)):
        if compound[i].isdigit():
            compoundnew.append(' ')
            compoundnew2.append(compound[i])
        else:
            compoundnew.append(compound[i])
            compoundnew2.append(' ')

    c2 = "".join(compoundnew)
    elements = c2.split()
    numelements = len(elements)
    c3 = "".join(compoundnew2)
    stoichiometry = c3.split()
    return elements, stoichiometry


# -------------------------------------
def split_formula_dict(compound):
    compoundnew = []
    compoundnew2 = []

    for i in range(len(compound)):
        if compound[i].isdigit():
            compoundnew.append(' ')
            compoundnew2.append(compound[i])
        else:
            compoundnew.append(compound[i])
            compoundnew2.append(' ')

    c2 = "".join(compoundnew)
    elements = c2.split()
    numelements = len(elements)
    c3 = "".join(compoundnew2)
    stoichiometry = c3.split()

    stoich_dict = {}
    for i in range(len(elements)):
        stoich_dict[elements[i]] = stoichiometry[i]

    return stoich_dict


# ------------------------------------
def format_nrelmatdb(readcsvfile, writecsvfile):
    df = pandas.read_csv(readcsvfile)
    compounds = df['#sortedformula'].tolist()

    for i in range(len(compounds)):
        comps = compounds[i].split()
        newcomps = []
        for comp in comps:
            trigger = 0

            for el in comp:
                if el.isdigit():
                    trigger = 1

            if trigger == 1:
                newcomps.append(comp)
            elif trigger == 0:
                newcomps.append(comp + '1')

        df.at[i, '#sortedformula'] = ''.join(newcomps)

    df = df.rename(columns={'#sortedformula': 'sortedformula'})
    df.to_csv(writecsvfile, index=False)


# ------------------------------------
def find_icsdphases(elementset):
    # element set is a list of elements
    trigger = 0
    compounds_found = []

    df = pandas.read_csv('icsdcomposition.csv')
    compounds = df['compound'].tolist()

    # search the database of ordered and stoichiometric compounds to find competing phases
    for compound in compounds:

        elements, stoich = split_formula(compound)

        if any(x not in elementset for x in elements):
            continue
        else:
            # if len(elements) > 1 and trigger == 0:
                # # print '----------------'
                # print('COMPETING PHASES FOUND IN ICSD')
                # # print '----------------'

            if len(elements) > 1:
                # print compound
                trigger = 1
                compounds_found.append(compound)

    if trigger == 0:
        print('DID NOT FIND ANY COMPETING PHASES IN NRELMATDB')

    return compounds_found


# -------------------------------------

def find_in_nrelmatdb(elementset):
    df1 = pandas.read_csv('nrelmatdb_1.csv')
    df2 = pandas.read_csv('nrelmatdb_2.csv')

    compounds1 = df1['sortedformula'].tolist()
    compounds2 = df2['sortedformula'].tolist()
    compounds = compounds1 + compounds2

    compounds_found = []
    trigger = 0

    for comp in compounds:
        compound = comp

        elements, stoich = split_formula(compound)

        if any(x not in elementset for x in elements):
            continue
        else:
            # if len(elements) > 1 and trigger == 0:
                # # print '----------------'
                # print('COMPETING PHASES FOUND IN NRELMATDB')
                # # print '----------------'

            if len(elements) > 1 and compound not in compounds_found:
                # print compound
                trigger = 1
                compounds_found.append(compound)

    if trigger == 0:
        print('DID NOT FIND ANY COMPETING PHASES IN NRELMATDB')

    return compounds_found


# --------------------------------------

def find_energy_nrelmatdb(elementset, df):
    # df1 = pandas.read_csv('nrelmatdb_1.csv')
    # compounds1 = df1['sortedformula'].tolist()
    # icsdnums1 = df1['icsdnum'].tolist()
    # totalenergies1 = df1['energyperatom'].tolist()

    # df2 = pandas.read_csv('nrelmatdb_2.csv')
    # compounds2 = df2['sortedformula'].tolist()
    # icsdnums2 = df2['icsdnum'].tolist()
    # totalenergies2 = df2['energyperatom'].tolist()

    compounds = df['sortedformula'].tolist()
    icsdnums = df['icsdnum'].tolist()
    totalenergies = df['energyperatom'].tolist()

    # data collected in dictionary energies
    energies = {}
    compounds_found = []

    trigger = 0

    for i in range(len(compounds)):
        compound = compounds[i]
        icsdnum = icsdnums[i]
        totalenergy = totalenergies[i]

        elements, stoich = split_formula(compound)

        if any(x not in elementset for x in elements):
            continue
        else:
            # if len(elements) > 1 and trigger == 0:
                # # print '----------------'
                # print('COMPETING PHASES FOUND IN NRELMATDB')
                # # print '----------------'

            if len(elements) > 1 and compound not in compounds_found:
                # print compound
                trigger = 1
                compounds_found.append(compound)
                energies[compound] = [[icsdnum, round(totalenergy, 3)]]

            if len(elements) > 1 and compound in compounds_found:
                trigger = 1
                newlist = list(energies[compound])
                newlist.append([icsdnum, round(totalenergy, 3)])
                energies[compound] = newlist

    if trigger == 0:
        print('DID NOT FIND ANY COMPETING PHASES IN NRELMATDB')

    return energies


# ----------------------------------------

def find_not_in_nrelmatdb(elementset):
    icsdcompounds = find_icsdphases(elementset)
    nrelmatdbcompounds = find_in_nrelmatdb(elementset)
    missing_compounds = []
    trigger = 0

    for comp in icsdcompounds:
        if comp not in nrelmatdbcompounds:
            if trigger == 0:
                print('FOUND MISSING PHASES')
                trigger = 1
            missing_compounds.append(comp)

    if trigger == 0:
        print('ALL PHASES ARE IN NRELMATDB')

    return missing_compounds


# ----------------------------------------

def create_input_DFT(elementset, df, chempot='fere', filepath=None):
    energies = find_energy_nrelmatdb(elementset, df)

    if len(energies) == 0:
        print('CANNOT PROCEED')
        return

    energies_added = {}
    compounds_added = []

    # finding the min energy structures for each composition
    energies_min = {}

    for key, values in energies.items():
        newlist = list(energies[key])
        sorted_newlist = sorted(newlist, key=itemgetter(1))
        energies_min[key] = sorted_newlist[0]

    if chempot == 'fere' or chempot == 'ferev2':
        fere_trigger = 1
    else:
        fere_trigger = 0

    # sanity check on the arguments
    if fere_trigger == 0 and len(elementset) != len(chempot):
        print('Number of elements in elementset and chempot do not match')
        return

    if fere_trigger == 0:
        numelement = 0
        for el in elementset:
            try:
                if chempot[el] <= 0: numelement += 1
            except:
                continue

        if numelement != len(elementset):
            print('Reference chemical potential of all elements in elemenset not found in chempot: CANNOT PROCEED')
            return

        chempot_match = 0
        for el in elementset:
            if el in chempot: chempot_match += 1
        if chempot_match != len(elementset):
            print('Chemical potential of all elements not provided: CANNOT PROCEED')
            return

    # assembling the reference chemical potentials    
    sorted_es = sorted(elementset)

    if fere_trigger == 0:
        reference_chempot = chempot

    if chempot == 'fere':
        reference_chempot = {}
        for el in elementset:
            try:
                reference_chempot[el] = fere_chempot[el]
            except:
                print('FERE value not available for element %s' % el)
                return

    if chempot == 'ferev2':
        reference_chempot = {}
        for el in elementset:
            try:
                reference_chempot[el] = ferev2_chempot[el]
            except:
                print('FEREv2 value not available for element %s' % el)
                return

    # writing input file
    # f = open(filepath,'w')
    out_strs = []
    out_strs.append(' '.join(sorted_es))
    out_strs.append(' '.join(f'{reference_chempot[es]:2.3f}' for es in sorted_es))

    for es in sorted_es:
        write_str = ''
        stoich_dict = split_formula_dict('%s1' % es)
        for es1 in sorted_es:
            if es1 in stoich_dict:
                write_str += '%s ' % stoich_dict[es]
            else:
                write_str += '0 '
        write_str += '%2.3f' % reference_chempot[es]
        out_strs.append(write_str)

    for key, value in energies_min.items():
        write_str = ''
        stoich_dict = split_formula_dict(key)
        for es in sorted_es:
            if es in stoich_dict:
                write_str += '%s ' % stoich_dict[es]
            else:
                write_str += '0 '

        write_str += '%2.3f' % value[1]
        out_strs.append(write_str)

    if filepath is not None:
        with open(filepath, 'w') as out:
            out.write('\n'.join(out_strs) + '\n')
    return out_strs
