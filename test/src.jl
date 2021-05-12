#=
test_src:
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code,
without *many* machine learning components
=#

##########################
# SOURCE TESTS #
##########################

""" Tests expression parsing. """
function test_expressions()
    model, x, y, z, a = test_model()
    expr = :((x, y, z) -> sum(x[i] for i=1:4) - y[1] * y[2] + z)
    simp_expr = :(x -> sum(5 .* x))
    f = functionify(expr)

    # Testing function evaluation
    res = Base.invokelatest(f, (ones(4), ones(2),5)...)
    @test res isa Float64
    res = Base.invokelatest(f, (x, y, z)...)
    @test res isa JuMP.GenericQuadExpr
    @test res == sum(x[1:4]) - y[1] * y[2] + z

    # Testing variable input parsing from expression
    @test vars_from_expr(expr, model) == [x,y,z]
    @test vars_from_expr(simp_expr, model) == [x]
    @test vars_from_expr(:((x, a) -> x[2] + a[2,2]),model) == [x, a]

    # Testing "flattening of expressions" for nonlinearization
    expr_vars = vars_from_expr(expr, model)
    @test OCT.get_var_ranges(expr_vars) == [(1:5),(6:8),9]
    @test OCT.zeroarray([(1:5),(6:8),9]) == [zeros(5), zeros(3), 0]
    flat_expr = :((x...) -> $(expr)([x[i] for i in $(OCT.get_var_ranges(expr_vars))]...))
    fn = functionify(flat_expr)
    @test Base.invokelatest(fn, [1,2,3,4,1,5,-6,-7,7]...) == Base.invokelatest(f, ([1,2,3,4,1], [5,-6,-7], 7)...)
    @test Base.invokelatest(fn, flat(expr_vars)...) == res

    # Testing proper mapping for expressions
    flatvars = flat([y[2], z, x[1:4]])
    vars = vars_from_expr(expr, model)
    @test OCT.get_varmap(vars, flatvars) == [(2,2), (3,0), (1,1), (1, 2), (1,3), (1,4)]
    @test OCT.get_datamap(vars, flatvars) == [7, 9, 1, 2, 3, 4]
    
    # Testing gradientify
    grad = gradientify(expr, vars_from_expr(expr, model))
    @test all(grad(ones(9)) .≈ [1, 1, 1, 1, 0, -1, -1, 0, 1])
    other_grad = gradientify(simp_expr, vars_from_expr(simp_expr, model))
    @test all(other_grad(ones(5)) .≈ 5 .* ones(5))
    @test all(grad(ones(9)) .≈ [1, 1, 1, 1, 0, -1, -1, 0, 1]) # in case of world age problems
    con = @constraint(model, Base.invokelatest(f, (x,y,z)...) >= 1) # also on JuMP constraints
    gradfn = gradientify(con, expr_vars)
    @test all(gradfn(ones(9)) .≈ grad(ones(9)))
    @test_throws OCTException gradientify(@constraint(model, [x[1] x[2];
                                                              x[3] x[4]] in PSDCone()), x[1:4])

    # Testing vars_from_constraint as well
    @test all([var in [x[1], x[2], x[3], x[4], y[1], y[2], z] for var in vars_from_constraint(con)])
    m = pool1(false)
    l_constrs, nl_constrs = classify_constraints(m)
    l_vars = vars_from_constraint.(l_constrs)
    @test length.(l_vars) == [4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    nl_vars = vars_from_constraint.(nl_constrs)
    @test length.(nl_vars) == [3,3,3,2]
    g_fns = gradientify.(nl_constrs, nl_vars)
    @test g_fns[1](ones(length(nl_vars[1]))) == [-1, 2, -8]
end

function test_variables()
    model, x, y, z, a  = test_model()

    # Testing getting variables
    varkeys = ["x[1]", x[1], :z, :x];
    vars = [x[1], x[1], z, x[:]];
    @test all(vars .==  fetch_variable(model, varkeys))
end

function test_bounds()
    model, x, y, z, a = test_model()

    # Bounds and fixing variables
    bounds = get_bounds(model);
    @test bounds[z] == [-30., Inf]
    bounds = Dict(:x => [-10,1], z => [-10, 10])
    bound!(model, bounds)
    @test get_bounds(model)[z] == [-10, 10]
    @test get_bounds(model)[x[3]] == [-5, 1]

    # Check infeasible bounds
    new_bound = x[4] => [-10,-6]
    @test_throws OCTException OCT.check_infeasible_bound(new_bound)
    @test_throws OCTException bound!(model, new_bound)

    # Check unbounds
    @test all(collect(values(get_unbounds(model)))[i] == [-2., Inf] for i = 1:4)

    # Check that JuMP does not set Inf bounds
    m = JuMP.Model()
    @variable(m, x[1:2])
    @variable(m, y >= 5)
    bound!(m, Dict(var => [-Inf, Inf] for var in all_variables(m)))
    @test !JuMP.has_lower_bound(x[1]) && JuMP.has_lower_bound(y) && !JuMP.has_upper_bound(x[2])
    bound!(m, Dict(var => [-10, 10] for var in all_variables(m)))
    @test isnothing(get_unbounds(m))
end

function test_sets()
    sets = [MOI.GreaterThan(2), MOI.EqualTo(0), MOI.SecondOrderCone(3), MOI.GeometricMeanCone(2), MOI.SOS1([1,2,3])]
    @test get_constant.(sets) == [2, 0, nothing, nothing, nothing]

    model, x, y, z, a = test_model()
    restrict_to_set(x[3], [1,2,3])
    @test length(all_variables(model)) == 18
end

function test_linearize()
    model, x, y, z, a = test_model()
    # Linearization of objective
    linearize_objective!(model);
    @objective(model, Min, x[3]^2)
    linearize_objective!(model);
    @test JuMP.objective_function(model) isa JuMP.VariableRef

    @constraint(model, sum(x[4]^2 + x[5]^2) <= z)
    @constraint(model, sum(y[:]) >= -2)
    # Separation of model constraints
    l_constrs, nl_constrs = classify_constraints(model)
    @test length(l_constrs) == 27 && length(nl_constrs) == 1
end

function test_nonlinearize(gm::GlobalModel = minlp(true))
    nonlinearize!(gm)
    set_optimizer(gm, CPLEX_SILENT)
    @test_throws ErrorException("The solver does not support nonlinear problems (i.e., NLobjective and NLconstraint).") optimize!(gm)
    @test true
end

function test_bbc()
    model, x, y, z, a = test_model()
    nl_constr = @constraint(model, sum(x[4]^2 + x[5]^2) <= z)
    expr = :((x, y, z) -> sum(x[i] for i=1:4) - y[1] * y[2] + z)

    # "Sanitizing" data
    inp = Dict(x[1] => 1., x[2] => 2, x[3] => 3, x[4] => 4, x[5] => 1, y[1] => 5, y[2] => -6, y[3] => -7, z => 7)
    inp_dict = Dict(string(key) => val for (key, val) in inp)
    inp_df = DataFrame(inp_dict)
    @test data_to_DataFrame(inp) == data_to_DataFrame(inp_dict) == data_to_DataFrame(inp_df) == inp_df
    @test data_to_Dict(inp_df, model) == data_to_Dict(inp, model) == data_to_Dict(inp_dict, model) == inp

    # Test bbl creation
    @test isnothing(functionify(nl_constr))
    @test functionify(expr) isa Function
    bbls = [BlackBoxClassifier(constraint = nl_constr, vars = [x[4], x[5], z], expr_vars = [x[4], x[5], z]),
        BlackBoxClassifier(constraint = expr, vars = flat([x[1:4], y[1:2], z]),
                         expr_vars = [x,y,z])]

    # Evaluation (scalar)
    # Quadratic (JuMP compatible) constraint
    @test evaluate(bbls[1], inp) == evaluate(bbls[1], inp_dict) == evaluate(bbls[1], inp_df) == -10.
    # Nonlinear expression
    @test evaluate(bbls[2], inp) == evaluate(bbls[2], inp_dict) == evaluate(bbls[2], inp_df)

    # Evaluation (vector)
    inp_df = DataFrame(-5 .+ 10 .*rand(3, size(inp_df,2)), string.(keys(inp)))
    inp_dict = data_to_Dict(inp_df, model)
    @test evaluate(bbls[1], inp_dict) == evaluate(bbls[1], inp_df) == inp_df[!, "z"] -
                                                    inp_df[!, "x[4]"].^2 - inp_df[!, "x[5]"].^2
    @test evaluate(bbls[2], inp_dict) == evaluate(bbls[2], inp_df)

    # bbl CHECKS
    bbl = bbls[1]

    # Check unbounded sampling
    @test_throws OCTException X = lh_sample(bbl, n_samples=100);

    # Check knn_sampling without previous samples
    @test_throws OCTException knn_sample(bbl, k=3)

    # Check evaluation of samples
    samples = DataFrame(randn(10, length(bbl.vars)),string.(bbl.vars))
    vals = bbl(samples);
    @test vals ≈ -1*samples[!, "x[4]"].^2 - samples[!, "x[5]"].^2 + samples[!, "z"]

    # Checks different kinds of sampling
    bound!(model, Dict(z => [-Inf, 10]))
    X_bound = boundary_sample(bbl);
    @test size(X_bound, 1) == 2^(length(bbl.vars)+1)
    @test_throws OCTException knn_sample(bbl, k=3)
    X_lh = lh_sample(bbl, lh_iterations=3);

    # Check sample_and_eval
    uniform_sample_and_eval!(bbl);

    # Sampling, learning and showing...
    learn_constraint!(bbl);

    # Check feasibility and accuracy
    @test 0 <= feasibility(bbl) <= 1
    @test 0 <= evaluate_accuracy(bbl) <= 1

    # Training a model
    jc = add_feas_constraints!(model, bbl.vars, bbl.learners[1]);
    @test true
end

""" Testing some IAI kwarging. """
function test_kwargs()
    # Classification kwargs first...
    sample_kwargs = Dict(:localsearch => false, 
                       :invalid_kwarg => :hello,
                       :ls_num_tree_restarts => 20)

    dict_fit = fit_classifier_kwargs(; sample_kwargs...)
    dict_fit2 = fit_classifier_kwargs(localsearch = false, invalid_kwarg = :hello,
                           ls_num_tree_restarts = 20)
    @test dict_fit == dict_fit2 == Dict(:sample_weight => :autobalance)

    dict_lnr = classifier_kwargs(; sample_kwargs...)
    dict_lnr2 = classifier_kwargs(localsearch = false, invalid_kwarg = :hello, ls_num_tree_restarts = 20)
    @test dict_lnr == dict_lnr2 == Dict(:localsearch => false, :ls_num_tree_restarts => 20)

    # Regression kwargs next...
    dict_fit = fit_regressor_kwargs(; sample_kwargs)
    @test dict_fit == Dict()

    dict_lnr = regressor_kwargs(; sample_kwargs...)
    dict_lnr2 = regressor_kwargs(localsearch = false, invalid_kwarg = :hello, ls_num_tree_restarts = 20)
    @test dict_lnr == dict_lnr2 == Dict(:localsearch => false, :ls_num_tree_restarts => 20)
end

""" Tests different kinds of regression. """
function test_regress()
    X = DataFrame(:x => 3*rand(100) .- 1, :y => 3*rand(100) .- 1);
    Y = Array(X[!,:x].^3 .* sin.(X[!,:y]));
    α0, α = u_regress(X, Y)
    β0, β = l_regress(X, Y)
    γ0, γ = ridge_regress(X, Y)
    lowers = β0 .+ Matrix(X) * β;
    uppers = α0 .+ Matrix(X) * α;
    best_fit = γ0 .+ Matrix(X) * γ;
    @test all(lowers .<= Y) && all(uppers .>= Y) && all(uppers .>= lowers)
    errors = [sum((lowers-Y).^2), sum((uppers-Y).^2), sum((best_fit-Y).^2)]
    @test errors[3] <= errors[1] && errors[3] <= errors[2]  

    X = DataFrame(:x => rand(100), :y => rand(100))
    Y = X[!,:y] - X[!,:x] .+ 0.1
    solver = CPLEX_SILENT
    β0, β = svm(Matrix(X), Y)
    predictions = Matrix(X) * β .+ β0 
    @test sum((predictions-Y).^2) <= 1e-10
end

""" Tests various ways to train a regressor. """
function test_bbr()
    gm = minlp(true)
    set_optimizer(gm, CPLEX_SILENT)
    uniform_sample_and_eval!(gm)
    bbr = gm.bbls[3]

    # Make sure to train and add bbcs, and forget for a while...
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    learn_constraint!(bbcs)
    @test all(evaluate_accuracy.(bbcs) .>= 0.95)
    for bbc in bbcs
        add_tree_constraints!(gm, bbc)
    end
    
    # Threshold training
    learn_constraint!(bbr, "upper" => 20.)
    lnr = bbr.learners[end]
    @test lnr isa IAI.OptimalTreeClassifier
    all_leaves = find_leaves(lnr)
    # Add a binary variable for each leaf
    feas_leaves =
        [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
    @test sort(-1 .*collect(keys(bbr.ul_data[end]))) == sort(feas_leaves) # upper bounding leaves have negative idxs
    @test bbr.thresholds[end] == ("upper" => 20.)
    
    # Check adding of upper bounding constraint to empty model
    types = JuMP.list_of_constraint_types(gm.model)
    init_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    init_variables = length(all_variables(gm))
    add_tree_constraints!(gm, bbr)
    @test sort(union(feas_leaves, [1])) == sort(abs.(collect(keys(bbr.mi_constraints))))
    @test sort(abs.(collect(keys(bbr.leaf_variables)))) == sort(feas_leaves)

    # Checking that correct numbers of constraints and variables are added
    types = JuMP.list_of_constraint_types(gm.model)
    final_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    final_variables = length(all_variables(gm.model))
    @test final_constraints == init_constraints + length(all_mi_constraints(bbr)) + length(bbr.leaf_variables)    
    @test final_variables == init_variables + length(bbr.leaf_variables) + 
                                length(bbcs[1].leaf_variables) + length(bbcs[2].leaf_variables)

    # Check clearing of all constraints and variables
    clear_tree_constraints!(gm, bbr)
    types = JuMP.list_of_constraint_types(gm.model)
    @test init_constraints == sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    @test init_variables == length(all_variables(gm))
    # Since all_variables(gm) doesn't count auxiliary variables...
    @test length(all_variables(gm.model)) == length(all_variables(gm)) +
            length(bbcs[1].leaf_variables) + length(bbcs[2].leaf_variables)     

    # Flat prediction training
    learn_constraint!(bbr, regression_sparsity = 0, max_depth = 2)
    lnr = bbr.learners[end]
    all_leaves = find_leaves(lnr)
    @test lnr isa IAI.OptimalTreeRegressor
    @test length(bbr.ul_data[end]) == length(all_leaves) * 2 # double since contains lower and upper data
    @test bbr.thresholds[end] == Pair("reg", nothing)

    # Full regression training
    learn_constraint!(bbr, max_depth = 0)
    lnr = bbr.learners[end]
    @test lnr isa IAI.OptimalTreeRegressor
    all_leaves = find_leaves(lnr)
    @test length(bbr.ul_data[end]) == length(all_leaves) * 2
    @test bbr.thresholds[end] == Pair("reg", nothing)

    # Lower regression training
    learn_constraint!(bbr, "lower" => 5.)
    @test all(sign.(collect(keys(bbr.ul_data[end]))) .== 1)
    @test bbr.thresholds[end] == ("lower" => 5.)

    # upperlower regression training (sets an upper bound, but lower bounds in leaves)
    learn_constraint!(bbr, "upperlower" => 10.)
    lnr = bbr.learners[end]
    @test sum(collect(keys(bbr.ul_data[end]))) == 0

    # Checking all possible update scenarios
    # Note: [1] => upper bounds, [2,3] => regressors, [4] => lower bounds
    update_tree_constraints!(gm, bbr, 1) # Updating nothing with single upper bound
    @test bbr.active_trees == Dict(1 => bbr.thresholds[1]) && 
        all(sign.(collect(keys(bbr.mi_constraints))) .== -1)
    @test isnothing(active_lower_tree(bbr))
    @test active_upper_tree(bbr) == 1

    clear_tree_constraints!(gm, bbr)
    update_tree_constraints!(gm, bbr, 2) # Updating nothing with single regressor
    @test all(sign.(collect(keys(bbr.leaf_variables))) .== 1) &&
            length(bbr.leaf_variables) + 1 == length(bbr.mi_constraints) &&
                bbr.active_trees == Dict(2 => bbr.thresholds[2])
    @test active_lower_tree(bbr) == 2 && active_upper_tree(bbr) == 2

    update_tree_constraints!(gm, bbr, 3) # Replacing regressor with a regressor
    # No upper and lower bounding, just regressor here
    @test all(sign.(collect(keys(bbr.leaf_variables))) .== 1) &&
            bbr.active_trees == Dict(3 => bbr.thresholds[3])
    @test active_lower_tree(bbr) == 3 && active_upper_tree(bbr) == 3

    update_tree_constraints!(gm, bbr, 4) # Replacing regressor with lower bound
    @test all(sign.(collect(keys(bbr.leaf_variables))) .== 1) && # since lower bounding
        length(bbr.leaf_variables) + 1 == length(bbr.mi_constraints) && 
            bbr.active_trees == Dict(4 => bbr.thresholds[4])
    @test active_lower_tree(bbr) == 4
    @test isnothing(active_upper_tree(bbr))

    update_tree_constraints!(gm, bbr, 1) # Adding upper bound to lower bound
    @test length(bbr.leaf_variables) + 2 == length(bbr.mi_constraints) && # checking leaf variables
        length(bbr.active_trees) == 2
    optimize!(gm)
    @test active_lower_tree(bbr) == 4 && active_upper_tree(bbr) == 1

    clear_tree_constraints!(gm, bbr)
    update_tree_constraints!(gm, bbr, 1)
    update_tree_constraints!(gm, bbr, 4) # Adding lower bound to upper bound
    @test length(bbr.leaf_variables) + 2 == length(bbr.mi_constraints) && # checking leaf variables
        length(bbr.active_trees) == 2
    optimize!(gm)
    @test active_lower_tree(bbr) == 4 && active_upper_tree(bbr) == 1

    update_tree_constraints!(gm, bbr, 5) # Replace separate u/l bounds with upperlower
    @test length(bbr.leaf_variables)*2 + 1 == length(bbr.mi_constraints) &&
        length(bbr.active_trees) == 1
    @test active_lower_tree(bbr) == 5 && active_upper_tree(bbr) == 5

    # Make sure that all solutions are the same. 
    update_tree_constraints!(gm, bbr, 1)
    update_tree_constraints!(gm, bbr, 4) 
    optimize!(gm)
    update_tree_constraints!(gm, bbr, 1)
    optimize!(gm)

    # Testing leaf sampling
    df = last_leaf_sample(bbr)
    active_tree_idxs = collect(keys(bbr.active_trees))
    @test sort(active_tree_idxs) == [1, 4]
    @test size(df, 1) == get_param(bbr, :n_samples)
    for bbc in bbcs
        df = last_leaf_sample(bbc)
        @test size(df, 1) == get_param(bbc, :n_samples)
    end
    update_tree_constraints!(gm, bbr, 2)
    @test_throws OptimizeNotCalled last_leaf_sample(bbcs[1])

    @test all(Array(gm.solution_history[:,"obj"]) .≈ gm.solution_history[1, "obj"])

    # Checking proper storage
    @test all(length.([bbr.ul_data, bbr.thresholds, bbr.learners, bbr.learner_kwargs]) .== 5)
    clear_tree_data!(bbr)
    @test all(length.([bbr.ul_data, bbr.thresholds, bbr.learners, bbr.learner_kwargs]) .== 0)
    @test !isempty(bbr.mi_constraints) && !isempty(bbr.leaf_variables)
    clear_tree_constraints!(gm, bbr)
    @test isempty(bbr.mi_constraints) && isempty(bbr.leaf_variables)
end

""" Tests basic functionalities in GMs. """
function test_basic_gm()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT)

    # Actually trying to optimize...
    find_bounds!(gm, all_bounds=true)
    uniform_sample_and_eval!(gm)

    learn_constraint!(gm)
    println("Approximation accuracies: ", evaluate_accuracy(gm))

    # Solving of model
    set_param(gm, :ignore_accuracy, true)
    add_tree_constraints!(gm)
    optimize!(gm)
    vals = solution(gm);
    init_leaves = [find_leaf_of_soln(bbl) for bbl in gm.bbls]
    @test all(init_leaves[i] in keys(gm.bbls[i].leaf_variables) for i=1:length(gm.bbls))
    println("X values: ", vals)
    println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))

    # Testing constraint addition and removal
    clear_tree_constraints!(gm) # Clears all bbl constraints
    @test !any(is_valid(gm.model, constraint) for constraint in all_mi_constraints(gm.bbls[2]))
    add_tree_constraints!(gm, gm.bbls[2])
    @test all(is_valid(gm.model, constraint) for constraint in all_mi_constraints(gm.bbls[2]))
    clear_tree_constraints!(gm);
    add_tree_constraints!(gm);
    clear_tree_constraints!(gm, gm.bbls[1])
    @test !any(is_valid(gm.model, constraint) for constraint in all_mi_constraints(gm.bbls[1]))
    clear_tree_constraints!(gm) # Finds and clears the one remaining bbl constraint.
    @test all([!is_valid(gm.model, constraint) for constraint in gm.bbls[1].mi_constraints])
    @test all([!is_valid(gm.model, var) for var in values(gm.bbls[1].leaf_variables)])

    # Saving fit for test_load_fits()
    # save_fit(gm)

    # Testing clearing all data
    clear_data!(gm)
    @test all([size(bbl.X, 1) == 0 for bbl in gm.bbls])
    @test all([length(bbl.learners) == 0 for bbl in gm.bbls])

    # Testing surveysolve, and add_infeasibility_cuts for 
    @test_throws OCTException surveysolve(gm) # no data error
    uniform_sample_and_eval!(gm)
    surveysolve(gm)
    update_leaf_vexity(gm.bbls[1])
    @test true
end

function test_convex_objective()
    gm = test_gqp()
    bbl = gm.bbls[1]
    set_param(bbl, :n_samples, 100)
    uniform_sample_and_eval!(gm)
    update_gradients(bbl, [1,2])
    @test !any(ismissing.(flat(Matrix(bbl.X[1:2,:])))) 
    update_gradients(bbl)
    hand_calcs = DataFrame(hcat([[6*x[1] + 2*x[2] + 1, 2*x[2] + 2*x[1] + 6] for x in eachrow(Matrix(bbl.X))]...)', string.(bbl.vars))
    @test all(all(Array(bbl.gradients[i,:]) .≈ Array(hand_calcs[i,:])) for i = 1:100)
    update_vexity(bbl)
    @test bbl.local_convexity == 1.0
    @test bbl.convex == true

    # "Learning" convex functions should not result in trees.
    learn_constraint!(bbl)
    @test isempty(bbl.learners)
    add_tree_constraints!(gm, bbl)
    @test length(bbl.mi_constraints[1]) >= 1
    @test isempty(bbl.leaf_variables)
    
    # Testing adding gradient cuts
    optimize!(gm)
    @test find_leaf_of_soln.(gm.bbls) == [1]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) >  get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test all([gm.cost[i] < gm.cost[i+1] for i= 1:length(gm.cost)-1]) 
    update_leaf_vexity(gm.bbls[1])
    @test gm.bbls[1].vexity[1][2] == 1.0
end

function test_data_driven()
    model, x, y, z, a = test_model()
    rand_data = DataFrame("d" => [1,2,3])
    @test_throws UndefVarError bound_to_data!(model, rand_data)
    add_variables_from_data!(model, rand_data)
    @test model[:d] isa JuMP.VariableRef
    bound_to_data!(model, rand_data)
    @test get_bounds(model)[model[:d]] == [1,3]

    # Testing more complex constraints from afpm_model
    gm = afpm_model()
    @test length(gm.vars) == 15
    @test isnothing(get_unbounds(gm))
end

function test_rfs()
    gm = minlp(true)
    init_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) 
                            for type in JuMP.list_of_constraint_types(gm.model))
    set_optimizer(gm, CPLEX_SILENT)
    uniform_sample_and_eval!(gm)
    bbr = gm.bbls[3]
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier
            learn_constraint!(bbl)
            add_tree_constraints!(gm, bbl)
        elseif bbl isa BlackBoxRegressor
            learn_constraint!(bbl, "rfreg" => nothing)
            add_tree_constraints!(gm, bbl)
        end
    end
    optimize!(gm)
    clear_tree_constraints!(gm)
    @test init_constraints == sum(length(all_constraints(gm.model, type[1], type[2])) 
                                    for type in JuMP.list_of_constraint_types(gm.model))
end

# Fox and rabbit nonlinear population dynamics 
# Predator prey model with logistic function from http://www.math.lsa.umich.edu/~rauch/256/F2Lab5.pdf
function test_linking()
    m = Model(CPLEX_SILENT)
    t = 100
    r = 0.2
    x1 = 0.6
    y1 = 0.5
    @variable(m, x[1:t] >= 0.001) # Note: Ipopt solution does not converge with an upper bound!!
    @variable(m, dx[1:t-1])
    @variable(m, y[1:t] >= 0.001)
    @variable(m, dy[1:t-1])
    @constraint(m, x[1] == x1)
    @constraint(m, y[1] == y1)
    @constraint(m, [i=2:t], x[i] == x[i-1] + dx[i-1])
    @constraint(m, [i=2:t], y[i] == y[i-1] + dy[i-1])

    # NL dynamics solution using Ipopt
    # @NLconstraint(m, [i=1:t-1], dx[i] == x[i]*(1-x[i]) - x[i]*y[i]/(x[i]+1/5))
    # @NLconstraint(m, [i=1:t-1], dy[i] == r*y[i]*(1-y[i]/x[i]))

    # GlobalModel representation
    set_upper_bound.(x, 1)
    set_upper_bound.(dx, 1)
    set_lower_bound.(dx, -1)
    set_upper_bound.(y, 1)
    set_upper_bound.(dy, 1)
    set_lower_bound.(dy, -1)
    gm = GlobalModel(model = m, name = "foxes_rabbits")
    add_nonlinear_constraint(gm, :((x, y) -> x[1]*(1-x[1]) -x[1]*y[1]/(x[1]+0.2)), vars = [x[1], y[1]], 
        dependent_var = dx[1], equality=true)
    add_nonlinear_constraint(gm, :((x, y) -> 0.2*y[1]*(1-y[1]/x[1])), vars = [x[1], y[1]], 
        dependent_var = dy[1], equality=true)
    for i = 2:t-1
        add_linked_constraint(gm, gm.bbls[1], [x[i], y[i]], dx[i])
        add_linked_constraint(gm, gm.bbls[2], [x[i], y[i]], dy[i])
    end
    uniform_sample_and_eval!(gm)
    # Usually would want to train the dynamics better, but for speed this is better!
    learn_constraint!(gm, max_depth = 3, ls_num_tree_restarts = 5, ls_num_hyper_restarts = 5)
    set_param(gm, :ignore_accuracy, true)
    add_tree_constraints!(gm)
    optimize!(gm)

    # using Plots
    # # Plotting temporal population data
    # plot(getvalue.(x), label = "Prey")
    # plot!(getvalue.(y), label = "Predators", xlabel = "Time", ylabel = "Normalized population")
    # # OR simultaneously in the population dimensions
    # plot(getvalue.(m[:x]), getvalue.(m[:y]), xlabel = "Prey", ylabel = "Predators", label = 1:t, legend = false)
    # # return true
end

# function test_oos()
    op = oos_params()
    gm = oos_gm!()
    m = gm.model
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    add_tree_constraints!(gm)
    @test_throws MOI.ResultIndexBoundsError optimize!(gm)

    add_relaxation_variables!(gm)
    relax_objective(gm)
    add_tree_constraints!(gm)
    optimize!(gm)

    # Printing results
    println("Orbit altitudes (km) : $(round.(getvalue.(m[:r_orbit]), sigdigits=5))")
    println("Satellite order: $(Int.(round.(getvalue.(m[:sat_order]))))")
    println("True anomalies (radians): $(round.(getvalue.(m[:ta]), sigdigits=3))")
    println("Orbital revolutions: $(round.(abs.(op.period_sat .* getvalue.(m[:ta]) ./ getvalue.(m[:dt_orbit])), sigdigits=5)))")
    println("Time for maneuvers (days): $(round.(getvalue.(m[:t_maneuver])./(24*3600), sigdigits=3))")
    println("Total mission time (years): $(sum(getvalue.(m[:t_maneuver]))/(24*3600*365))")
# end

    
# test_expressions()

# test_variables()

# test_bounds()

# test_sets()

# test_linearize()

# test_nonlinearize()

# test_bbc()

# test_kwargs()

# test_regress()

# test_bbr()

# test_basic_gm()

# test_convex_objective()

# test_data_driven()

# test_linking()

# test_oos()
