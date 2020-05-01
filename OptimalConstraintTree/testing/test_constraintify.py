import numpy as np
from gpkit import Variable, Model
from gpkit.small_scripts import mag
from gpfit.fit import fit

import unittest

from gpkitmodels.SP.SimPleAC.SimPleAC_mission import *
from interpretableai import iai
import pickle

from OptimalConstraintTree.constraint_tree import ConstraintTree
from OptimalConstraintTree.global_model import GlobalModel
from OptimalConstraintTree.sample import gen_X
from OptimalConstraintTree.tools import (find_signomials, prep_SimPleAC,
                                        get_varkeys, get_bounds, \
                                        constraints_from_bounds, \
                                        constraint_from_gpfit, HiddenPrints)
from OptimalConstraintTree.testing.run_tests import run_tests

class TestConstraintify(unittest.TestCase):
    def test_monomials_from_pwl_data(self):
        pwlDict = {1: [1, [2,3,4]],
                   2: [5, [6,7,8]],
                   3: [9, [10,11,12]]}
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        ivars = [a, b*c**0.35, c]
        dvar = a*b
        constraintDict = ConstraintTree.pwl_constraintify(pwlDict, dvar, ivars)
        test_constr = np.exp(1)*ivars[0]**2*ivars[1]**3*ivars[2]**4 <= dvar
        self.assertEqual(test_constr.as_hmapslt1({}),
                         constraintDict[1][0].as_hmapslt1({}))

    def test_ConstraintTree_sp_constraints(self):
        """
        Tests ConstraintTree generation from SP constraints
        """
        m, basis = prep_SimPleAC()

        # Identify signomial constraints
        sp_constraints = find_signomials(m)
        sp_variables = get_varkeys(sp_constraints)
        #TODO: complete

    def test_SimPleAC_with_treeconstraint(self):
        m, basis = prep_SimPleAC()
        basesol = m.localsolve(verbosity=0)

        # Now replacing the drag model with a learner...
        constraints = [c for c in m.flat()]
        del constraints[-12:-8]
        lnr = iai.read_json("data/solar_airfoil_lnr.json")
        subs = m.substitutions.copy()
        for i in range(len(m['C_{D_{wpar}}'])):
            basis = {m['Re'][i].key: 1.5e6,
                     m['\\tau'].key:0.12}
            dvar = m['C_{D_{wpar}}'][i]
            ivars = [m['Re'][i],
                     m['\\tau'],
                     m['C_L'][i]]
            bounds = {
                m['Re'][i].key: [5e5,3e6],
                m['\\tau'].key: [0.08, 0.23],
                m['C_L'][i].key: [0.33, 2.0],
            }
            ct = ConstraintTree(lnr, dvar, ivars, basis=basis)
            constraints.append(ct)
            constraints.append(constraints_from_bounds(bounds, m))
        gm = GlobalModel(m.cost, constraints, subs)
        sol = gm.solve(verbosity=0)

    def test_SimPleAC_with_surrogate_tree(self):
        m, basis = prep_SimPleAC()
        # Replicate GP model with new models
        basesol = m.localsolve(verbosity=0)
        ivars = [m[var] for var in list(basis.keys())]
        dvar = Variable("Total cost", "N", "fuel and time cost")

        # Fitting GPfit model
        solns = pickle.load(open("data/SimPleAC.sol", "rb"))
        subs = pickle.load(open("data/SimPleAC.subs", "rb"))
        X = gen_X(subs, basis)
        Y = [mag(soln['cost'] / basesol['cost']) for soln in solns]
        with HiddenPrints():
            cstrt, rms = fit(np.log(np.transpose(X)), np.log(Y), 4, 'SMA')

        basis[dvar.key] = basesol['cost']*dvar.units
        fit_constraint = constraint_from_gpfit(cstrt, dvar, ivars, basis)
        basis.pop(dvar.key)
        m = Model(dvar, [fit_constraint], basis)
        fitsol = m.solve(verbosity=0, reltol=1e-6)
        self.assertAlmostEqual(fitsol['cost']/basesol['cost'], 1, places=2)

        # Now with trees
        lnr = iai.read_json("data/SimPleAC_lnr.json")
        basis[dvar.key] =  basesol['cost']*dvar.units
        ct = ConstraintTree(lnr, dvar, ivars, basis=basis)
        del basis[dvar.key]
        bounds = pickle.load(open("data/SimPleAC.bounds", "rb"))
        bounding_constraints = constraints_from_bounds(bounds, m)
        gm = GlobalModel(dvar, [bounding_constraints, ct], basis)
        sol = gm.solve(verbosity=0)
        self.assertAlmostEqual(sol['cost']/basesol['cost'], 1, places=2)

TESTS = [TestConstraintify]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()