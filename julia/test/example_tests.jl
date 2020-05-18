using DataFrames
using Gurobi, JuMP
using LatinHypercubeSampling
using Test

include("examples.jl")
include("../src/constraintify.jl")
include("../src/solve.jl")

"""
Set of examples for which to test different examples.
"""

function example_constraint_fit(fn_model)
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    nsamples = 1000;
    ndims = size(fn_model.lbs, 1)
    plan, _ = LHCoptim(nsamples, ndims, 1);
    X = scaleLHC(plan,[(fnm.lbs[i], fnm.ubs[i]) for i=1:3]);
end

function example_naive_fit(fnm)
    """
    Fits a provided function_model.
    Arguments:
        fnm:: instance of function_model
    """
    nsamples = 1000;
    ndims = 3;
    # Code for initial tree training
    plan, _ = LHCoptim(nsamples, ndims, 1);
    X = scaleLHC(plan,[(fnm.lbs[i], fnm.ubs[i]) for i=1:3]);
    lnr = base_otr()
    for i = 1:size(fnm.constr, 1)
        Y = [-fnm.constr[i](X[j,:]) for j=1:nsamples];
        IAI.fit!(lnr, X, Y)
        IAI.write_json(lnr, "data/" + name + "constraint" + string(i) + "_reg.json")
    end
    Y = [fnm.obj(X[j,:]) for j=1:nsamples];
    IAI.fit!(lnr, X, Y)
    IAI.write_json("data/" + name + "_objective.json", lnr)
end

# function example_naive_solve(fnm)
#     # Creating the model
#     constr = IAI.read_json("data/example1_constraint_naive.json")
#     objectivefn = IAI.read_json("data/example1_objective_naive.json")
#     vks = [Symbol("x",i) for i=1:3];
#     m = Model(solver=GurobiSolver());
#     @variable(m, x[1:3])
#     @variable(m, obj)
#     @objective(m, Min, obj)
#     add_mio_constraints(constr, m, x, 0, vks, 1000000);
#     add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
#     constraints_from_bounds(m, x, fnm.lbs, fnm.ubs);
#     status = solve(m)
#     println("Solved minimum: ", getvalue(obj))
#     println("Known global bound: ", -147-2/3)
#     println("X values: ", getvalue(x))
#     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
# end

# function example1_infeas()
#     nsamples = 1000;
#     ndims = 3;
#     # Code for initial tree training
#     plan, _ = LHCoptim(nsamples, ndims, 1);
#     X = scaleLHC(plan,[(fnm.lbs[i], fnm.ubs[i]) for i=1:3]);
#     # Assuming the objective has already been trained...
#     lnr = base_otr()
#     name = "example1"
#     feasTrees = learn_constraints(lnr, constraints, X, name=name)
#
#     # Creating the model
#     constr = IAI.read_json("data/example1_constraint_infeas.json")
#     objectivefn = IAI.read_json("data/example1_objective_naive.json")
#     vks = [Symbol("x",i) for i=1:3];
#     m = Model(solver=GurobiSolver());
#     @variable(m, x[1:3])
#     @variable(m, obj)
#     @objective(m, Min, obj)
#     add_feas_constraints(constr, m, x, vks, 1000);
#     add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
#     constraints_from_bounds(m, x, fnm.lbs, fnm.ubs);
#     status = solve(m)
#     println("Solved minimum: ", getvalue(obj))
#     println("Known global bound: ", -147-2/3)
#     println("X values: ", getvalue(x))
#     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
# end



function test_import_sagebenchmark()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    idxs = 1:25
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = import_sagebenchmark(idx)
    end
    return true
end

@test test_import_sagebenchmark()

@test example_

# @test example_naive_fit(example2)

# @test example1_infeas()
