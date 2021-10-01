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

def run_stability(inputs, phase):
    """
    Run stability analysis

    Arguments:
    inputs -- list of strings containing chemical potentials and total energies
    phase -- the structure of interest for phase stability     

    """
    trigger_phase_found = 0

    A, b, els, stoich = read_input(inputs)

    # f = open(filewrite,'w')

    h = Hrep(A, b)

    for is_stable in range(len(A)):

        # print '----------------------------------------------'

        borders = []
        comp_phases = []

        for gen in h.generators:
            if abs(dot(A[is_stable], gen) - b[is_stable]) < 0.00001:
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

        # if cmpdnew == phasenew:
        # print cmpd+'\n'
        # f.write(cmpd+'\n')

        if len(borders) == 0:
            if cmpdnew == phasenew:
                # print 'UNSTABLE\n'
                # f.write('UNSTABLE\n')
                return 'UNSTABLE'
                trigger_phase_found = 1
        else:
            if cmpdnew == phasenew:
                return 'STABLE'
                # print 'STABLE\n'
                # f.write('STABLE\n')
                trigger_phase_found = 1

            for border in borders:
                pom = []
                for i in range(len(A)):
                    if abs(dot(A[i], border) - b[i]) < 0.00001:
                        pom.append(stoich[i])
                comp_phases.append(pom)

            header = 'vertex '

            for i in range(len(els)):
                header = header + 'dmu_%s ' % els[i]
            header = header + 'competing_phases\n'

            # if cmpdnew == phasenew:
            # print header
            # f.write(header)

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
                    write_str = write_str[:-1] + '\n'
                    # print write_str
                    # f.write(write_str)

    # f.close()

    if trigger_phase_found == 0:
        print('DID NOT FIND PHASE: Please include all elements in the specifying the formula e.g. Te1Zn1Cd0')

    return


######################################

def run_stability_hull(inputs, phase, B):
    """
    Run stability analysis

    Arguments:
    filename -- input file containing chemical potentials and total energies
    phase -- the structure of interest for phase stability
    B -- list of formation energies of all phases, read from input file ('input_')

    """
    trigger_phase_found = 0

    A, b, els, stoich = read_input(inputs)

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

        # if cmpdnew == phasenew:
        # print cmpd+'\n'
        # f.write(cmpd+'\n')

        if len(borders) == 0:
            if cmpdnew == phasenew:
                # print 'UNSTABLE\n'
                # f.write('UNSTABLE\n')
                return 'UNSTABLE'
                trigger_phase_found = 1
        else:
            if cmpdnew == phasenew:
                return 'STABLE'
                # print 'STABLE\n'
                # f.write('STABLE\n')
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
            header = header + 'competing_phases\n'

            # if cmpdnew == phasenew:
            # print header
            # f.write(header)

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
                    write_str = write_str[:-1] + '\n'
                    # print write_str
                    # f.write(write_str)

    # f.close()

    if trigger_phase_found == 0:
        print('DID NOT FIND PHASE: Please include all elements in the specifying the formula e.g. Te1Zn1Cd0')

    return
