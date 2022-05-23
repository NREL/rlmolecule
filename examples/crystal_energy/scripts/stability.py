from copy import deepcopy

from numpy import *
from polyhedron import Hrep


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

def read_input(inputs):
    """
    Reads input file 

    Arguments:
    filename -- path to the input file containing chemical potentials and total energies

    """
    els = inputs[0]
    n = len(inputs[0].split())

    lines = inputs[2:]

    points = zeros([len(lines), n + 1])
    for i in range(len(lines)):
        points[i] = array(lines[i].split(), dtype=float64)

    mus = deepcopy(points)[:n, n]

    stoich = [deepcopy(x[:-1]) for x in points]

    for x in points:
        x[:n] = x[:n] / sum(x[:n])

    for x in points:
        x[n] = x[n] - dot(x[:n], mus)

    return points[:, :-1], points[:, -1], els.split(), stoich


######################################

def run_stability(inputs, phase, B=None, out_file=None):
    """
    Run stability analysis

    Arguments:
    inputs -- list of strings containing chemical potentials and total energies
    phase -- the structure of interest for phase stability     
    out_file -- write the results of the stability analysis.
        If structure is unstable, the output file will instead just have "UNSTABLE" in it

    """
    trigger_phase_found = 0

    A, b, els, stoich = read_input(inputs)
    B = b if B is None else B

    out_strs = []
    return_val = None

    h = Hrep(A, B)

    for is_stable in range(len(A)):

        # print '----------------------------------------------'

        borders = []
        comp_phases = []

        for gen in h.generators:
            if abs(dot(A[is_stable], gen) - B[is_stable]) < 0.00001:
                borders.append(gen)

        cmpd = ''
        for i in range(len(els)):
            cmpd = cmpd + els[i]
            cmpd = cmpd + str(int(stoich[is_stable][i]))

        el1, stoic1 = split_formula(phase)
        el2, stoic2 = split_formula(cmpd)

        el1, stoic1 = (list(t) for t in zip(*sorted(zip(el1, stoic1))))
        el2, stoic2 = (list(t) for t in zip(*sorted(zip(el2, stoic2))))

        phase_dummy = [el1[i] + str(stoic1[i]) for i in range(len(el1))]
        phasenew = ''.join(phase_dummy)

        cmpd_dummy = [el2[i] + str(stoic2[i]) for i in range(len(el2))]
        cmpdnew = ''.join(cmpd_dummy)

        if cmpdnew == phasenew:
            out_strs.append(cmpd)

        if len(borders) == 0:
            if cmpdnew == phasenew:
                out_strs.append('UNSTABLE')
                return_val = 'UNSTABLE' if return_val is None else return_val
                #return 'UNSTABLE'
                trigger_phase_found = 1
        else:
            if cmpdnew == phasenew:
                out_strs.append('STABLE')
                return_val = 'STABLE' if return_val is None else return_val
                #return 'STABLE'
                trigger_phase_found = 1

            for border in borders:
                pom = []
                for i in range(len(A)):
                    if abs(dot(A[i], border) - B[i]) < 0.00001:
                        pom.append(stoich[i])
                comp_phases.append(pom)

            header = 'vertex '

            for i in range(len(els)):
                header = header + 'dmu_%s ' % els[i]
            header = header + 'competing_phases'

            if cmpdnew == phasenew:
                out_strs.append(header)

            for i in range(len(borders)):

                bord = borders[i]

                write_str = ''

                for j in range(len(comp_phases[i])):
                    comp = comp_phases[i][j]

                    if j == 0:
                        out_string = 'V%s ' % (i + 1)

                        for kk in range(len(bord)):
                            out_string = out_string + '%2.3f ' % (bord[kk])

                        for kk in range(len(comp)):
                            out_string = out_string + els[kk] + str(int(comp[kk]))

                        out_string = out_string
                        if cmpdnew == phasenew:
                            # print out_string + ',',
                            write_str = write_str + out_string + ','

                    else:
                        out_string = ''

                        for kk in range(len(comp)):
                            out_string = out_string + els[kk] + str(int(comp[kk]))

                        out_string = out_string[:]

                        if cmpdnew == phasenew:
                            # print out_string+',',
                            write_str = write_str + out_string + ','

                if cmpdnew == phasenew:
                    write_str = write_str[:-1]
                    out_strs.append(write_str)

    if out_file is not None:
        print(f"writing {out_file}")
        with open(out_file, 'w') as out:
            out.write('\n'.join(out_strs) + '\n')

    if trigger_phase_found == 0:
        print('DID NOT FIND PHASE: Please include all elements in the specifying the formula e.g. Te1Zn1Cd0')

    #return
    return return_val

