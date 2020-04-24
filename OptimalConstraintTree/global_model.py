import numpy as np

from gpkit import Model
from gpkit.exceptions import InvalidGPConstraint
from gpkit.small_scripts import mag

from OptimalConstraintTree.constraint_tree import ConstraintTree

class GlobalModel(Model):
    """ Extends the GPkit Model class to be able to accommodate
    constraint trees.

    Arguments
    ---------
    cost : Posynomial

    constraints : ConstraintSet or list of constraints or ConstraintTrees.

    substitutions : dict
        Note: Do not forget substitutions required for ConstraintTrees and
        GP object fits.
    """
    sps = None

    def __init__(self, cost, constraints, *args, **kwargs):
        self.cost = cost
        self.trees = [c for c in constraints if isinstance(c, ConstraintTree)]
        self.sp_constraints = [c for c in constraints if not isinstance(c, ConstraintTree)]
        # self.treevars = set([[tree.dvar, ivar for ivar in tree.ivars]
        #                     for tree in self.trees])
        for key, value in kwargs.items():
            if key == 'solve_type':
                for tree in self.trees:
                    tree.solve_type = value
                tree.setup()
        print(cost)
        self.sp_model = Model(cost, self.sp_constraints, *args, **kwargs)

    def solve(self, verbosity=0, reltol=1e-3, x0=None):
        prev_cost = np.inf
        new_cost = 1e30
        if x0:
            xi = x0.copy()
        else:
            if verbosity >= 2:
                print("Generating initial first guess.")
            xi = self.sp_model.debug(verbosity=0)
        self.sps = []
        while prev_cost/new_cost - 1 >= reltol:
            constraints = self.sp_constraints.copy()
            for tree in self.trees:
                constraints.extend(tree.get_leaf_constraints(xi))
            self.sps.append(Model(self.cost, constraints, self.sp_model.substitutions))
            try:
                xi = self.sps[-1].solve(verbosity=verbosity)
            except InvalidGPConstraint:
                xi = self.sps[-1].localsolve(verbosity=verbosity)
            prev_cost = new_cost
            new_cost = mag(xi['cost'])
        return xi
