@with_kw mutable struct GlobalModel
"""
Contains all required info to be able to generate a global optimization problem.
NOTE: proper construction is to add constraints as BBFs.
"""
    model::JuMP.Model                                                     # JuMP model
    name::Union{Symbol, String} = "Model"                                 # Example name
    fns::Array{BlackBoxFunction} = Array{BlackBoxFunction}[]              # Black box (>/= 0) functions
    nl_constrs::Array = []                                                # Nonlinear constraints
    vars::Array{VariableRef} = JuMP.all_variables(model)                  # JuMP variables
end

function (gm::GlobalModel)(name::Union{String, Int64})
    """ Calls a BlackBoxFunction in GlobalModel by name. """
    fn_names = getfield.(gm.fns, :name);
    fns = gm.fns[findall(x -> x == name, fn_names)];
    if length(fns) == 1
        return fns[1]
    elseif length(fns) == 0
        @warn("No constraint with name ", name)
        return
    else
        @warn("Multiple constraints with name ", name)
        return fns
    end
end

function JuMP.all_variables(gm::GlobalModel)
    """ Extends JuMP.all_variables to GlobalModels,
        and makes sure that the variables are updated. """
    gm.vars = JuMP.all_variables(gm.model)
    return gm.vars
end

function JuMP.set_optimizer(gm::GlobalModel, optimizer_factory)
    """ Extends JuMP.set_optimizer to GlobalModels. """
    set_optimizer(gm.model, optimizer_factory)
end

function JuMP.optimize!(gm::GlobalModel)
    """ Extends JuMP.optimize! to GlobalModels. """
    optimize!(gm.model)
end

function get_bounds(model::Union{GlobalModel, JuMP.Model})
    """ Returns bounds of all variables from a JuMP.Model. """
    all_vars = all_variables(model)
    return get_bounds(all_vars)
end

function check_bounds(bounds::Dict)
    """ Checks outer-boundedness. """
    if any(isinf.(Iterators.flatten(values(bounds))))
        throw(OCTException("Unbounded variables in model!"))
    else
        return
    end
end

function get_max(a, b)
    return maximum([a,b])
end

function get_min(a,b)
    return minimum([a,b])
end

function check_infeasible_bounds(model::Union{GlobalModel, JuMP.Model}, bounds::Dict)
    all_bounds = get_bounds(model);
    lbs_model = Dict(key => minimum(value) for (key, value) in all_bounds)
    ubs_model = Dict(key => maximum(value) for (key, value) in all_bounds)
    lbs_bounds = Dict(key => minimum(value) for (key, value) in bounds)
    ubs_bounds = Dict(key => maximum(value) for (key, value) in bounds)
    nlbs = merge(get_max, lbs_model, lbs_bounds)
    nubs = merge(get_min, ubs_model, ubs_bounds)
    if any([nlbs[var] .> nubs[var] for var in keys(nlbs)])
        throw(OCTException("Infeasible bounds."))
    end
    return
end

function add_constraint(gm::GlobalModel,
                     constraint::Union{ScalarConstraint, JuMP.NonlinearExpression};
                     vars::Union{Nothing, Array{JuMP.VariableRef}} = nothing,
                     equality::Bool = false)
""" Adds a new nonlinear constraint to Global Model.
    Note: Linear constraints should be added directly to gm.Model. """
    bbf_vars = []
    if isnothing(vars)
        bbf_vars = JuMP.all_variables(gm.model)
    else
        bbf_vars = vars
    end
    if constraint isa JuMP.ScalarConstraint
        con = JuMP.add_constraint(gm.model, constraint)
        new_fn = BlackBoxFunction(constraint = nl_expr, vars = vars)
        push!(gm.fns, new_fn)
        push!(gm.nl_constrs, con)
    elseif constraint isa JuMP.NonlinearExpression
        if equality
            con = @NLconstraint(gm.model, constraint == 0)
            new_fn = BlackBoxFunction(constraint = con, vars = vars, equality = equality)
            push!(gm.fns, new_fn)
            push!(gm.nl_constrs, con)
        else
            con = @NLconstraint(gm.model, constraint >= 0)
            new_fn = BlackBoxFunction(constraint = con, vars = vars, equality = equality)
            push!(gm.fns, new_fn)
            push!(gm.nl_constrs, con)
        end
    end
end

function bound!(model::Union{GlobalModel, JuMP.Model},
                bounds::Dict)
    """Adds outer bounds to JuMP Model from dictionary of data. """
    if model isa GlobalModel
        bound!(model.model, bounds)
        return
    end
    check_infeasible_bounds(model, bounds)
    for (key, value) in bounds
        @assert value isa Array && length(value) == 2
        var = fetch_variable(model, key);
        if var isa Array # make sure all elements are bounded.
            for v in var
                bound!(model, Dict(v => value))
            end
        else
            if JuMP.has_lower_bound(var) && JuMP.lower_bound(var) <= minimum(value)
                set_lower_bound(var, minimum(value))
            elseif !JuMP.has_lower_bound(var)
                set_lower_bound(var, minimum(value))
            end
            if JuMP.has_upper_bound(var) && JuMP.upper_bound(var) >= maximum(value)
                set_upper_bound(var, maximum(value))
            else !JuMP.has_upper_bound(var)
                set_upper_bound(var, maximum(value))
            end
        end
    end
    return
end

function classify_constraints(model::Union{GlobalModel, JuMP.Model})
    """Separates and returns linear and nonlinear constraints in a model. """
    jump_model = model
    if model isa GlobalModel
        jump_model = model.model
    end
    all_types = list_of_constraint_types(jump_model)
    nl_constrs = [];
    l_constrs = [];
    l_vartypes = [JuMP.VariableRef, JuMP.GenericAffExpr{Float64, VariableRef}]
    l_constypes = [MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}]
    for (vartype, constype) in all_types
        constrs_of_type = JuMP.all_constraints(jump_model, vartype, constype)
        if any(vartype .== l_vartypes) && any(constype .== l_constypes)
            append!(l_constrs, constrs_of_type)
        else
            append!(nl_constrs, constrs_of_type)
        end
    end
    if !isnothing(jump_model.nlp_data)
        append!(nl_constrs, jump_model.nlp_data.nlconstr)
    end
    if model isa GlobalModel
        append!(model.nl_constrs, nl_constrs)
    end
    return l_constrs, nl_constrs
end

function feasibility(bbf::Union{GlobalModel, Array{BlackBoxFunction}, BlackBoxFunction})
    """ Returns the feasibility of data points in a BBF or GM. """
    if isa(bbf, BlackBoxFunction)
        return bbf.feas_ratio
    elseif isa(bbf, Array{BlackBoxFunction})
        return [feasibility(fn) for fn in bbf]
    else
        return [feasibility(fn) for fn in bbf.fns]
    end
end

function accuracy(bbf::Union{GlobalModel, BlackBoxFunction})
    """ Returns the accuracy of learners in a BBF or GM. """
    if isa(bbf, BlackBoxFunction)
        if bbf.feas_ratio in [1., 0]
            @warn(string("Accuracy of BlackBoxFunction ", bbf.name, " is tautological."))
            return 1.
        elseif isempty(bbf.learners)
            throw(OCTException(string("BlackBoxFunction ", bbf.name, " has not been trained yet.")))
        else
            return bbf.accuracies[end]
        end
    else
        return [accuracy(fn) for fn in bbf.fns]
    end
end

function lh_sample(bbf::BlackBoxFunction; iterations::Int64 = 3,
                n_samples::Int64 = 1000)
"""
Uniformly Latin Hypercube samples the variables of GlobalModel, as long as all
lbs and ubs are defined.
"""
   bounds = get_bounds(bbf.vars)
   check_bounds(bounds)
   n_dims = length(bbf.vars)
   plan, _ = LHCoptim(n_samples, n_dims, iterations);
   X = scaleLHC(plan,[(minimum(bounds[var]), maximum(bounds[var])) for var in bbf.vars]);
   return DataFrame(X, string.(bbf.vars))
end

function choose(large::Int64, small::Int64)
    return Int64(factorial(big(large)) / (factorial(big(large-small))*factorial(big(small))))
end

function boundary_sample(bbf::BlackBoxFunction; fraction::Float64 = 0.5)
""" *Smartly* samples the constraint along the variable boundaries.
    NOTE: Because we are sampling symmetrically for lower and upper bounds,
    the choose coefficient has to be less than ceil(half of number of dims). """
    n_vars = length(bbf.vars);
    vks = string.(bbf.vars);
    bounds = get_bounds(bbf.vars);
    check_bounds(bounds);
    lbs = DataFrame(Dict(string(key) => minimum(value) for (key, value) in bounds))
    ubs = DataFrame(Dict(string(key) => maximum(value) for (key, value) in bounds))
    n_comb = sum(choose(n_vars, i) for i=0:n_vars);
    nX = DataFrame([Float64 for i in vks], vks)
    sample_indices = [];
    if n_comb >= fraction*bbf.n_samples
        @warn("Can't exhaustively sample the boundary of Constraint " * string(bbf.name) * ".")
        n_comb = 2*n_vars+2; # Everything is double because we choose min's and max's
        choosing = 1;
        while n_comb <= fraction*bbf.n_samples
            choosing = choosing + 1;
            n_comb += 2*choose(n_vars, choosing);
        end
        choosing = choosing - 1; # Determined maximum 'choose' coefficient
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:choosing); # Choose 1 and above
    else
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:n_vars); # Choose 1 and above
    end
    for i in sample_indices
        lbscopy = copy(lbs); ubscopy = copy(ubs);
        lbscopy[:, vks[i]] = ubscopy[:, vks[i]];
        append!(nX, lbscopy);
        lbscopy = copy(lbs); ubscopy = copy(ubs);
        ubscopy[:, vks[i]] = lbscopy[:, vks[i]];
        append!(nX, ubscopy);
    end
    return nX
end

function knn_sample(bbf::BlackBoxFunction; k::Int64 = 15)
    """ Does KNN and interval arithmetic based sampling once there is at least one feasible
        sample to a BlackBoxFunction. """
    if bbf.feas_ratio == 0. || bbf.feas_ratio == 1.0
        throw(OCTException("Constraint " * string(bbf.name) * " must have at least one feasible or
                            infeasible sample to be KNN-sampled!"))
    end
    vks = string.(bbf.vars)
    df = DataFrame([Float64 for i in vks], vks)
    build_knn_tree(bbf);
    idxs, dists = find_knn(bbf, k=k);
    positives = findall(x -> x .>= 0 , bbf.Y);
    feas_class = classify_patches(bbf, idxs);
    for center_node in positives # This loop is for making sure that every possible root is sampled only once.
        if feas_class[center_node] == "mixed"
            nodes = [idx for idx in idxs[center_node] if bbf.Y[idx] < 0];
            push!(nodes, center_node)
            np = secant_method(bbf.X[nodes, :], bbf.Y[nodes, :])
            append!(df, np);
        end
    end
    return df
end

function sample_and_eval!(bbf::Union{BlackBoxFunction, Array{BlackBoxFunction}};
                          n_samples:: Union{Int64, Nothing} = nothing,
                          boundary_fraction::Float64 = 0.5,
                          iterations::Int64 = 3)
    """ Samples and evaluates BlackBoxFunction, with n_samples new samples.
    Arguments
    n_samples: number of samples, overwrites bbf.n_samples.
    boundary_fraction: maximum ratio of boundary samples
    iterations: number of GA populations for LHC sampling (0 is a random LH.)
    ratio:
    If there is an optimized gp, ratio*n_samples is how many random LHC samples are generated
    for prediction from GP. """
    if isa(bbf, Array{BlackBoxFunction})
        for fn in bbf
            sample_and_eval!(fn, n_samples = n_samples, boundary_fraction = boundary_fraction,
                             iterations = iterations);
        end
        return
    end
    if !isnothing(n_samples)
        bbf.n_samples = n_samples
    end
    vks = string.(bbf.vars)
    n_dims = length(vks);
    if size(bbf.X,1) == 0 # If we don't have data yet, uniform and boundary sample.
       df = boundary_sample(bbf, fraction = boundary_fraction)
       eval!(bbf, df)
       df = lh_sample(bbf, iterations = iterations, n_samples = bbf.n_samples - size(df, 1))
       eval!(bbf, df);
    elseif bbf.feas_ratio == 1.0
        @warn(string(bbf.name) * " was not KNN sampled since it has " * string(bbf.feas_ratio) * " feasibility.")
    elseif bbf.feas_ratio == 0.0
        throw(OCTException(string(bbf.name) * " was not KNN sampled since it has " * string(bbf.feas_ratio) * " feasibility.
                            Please find at least one feasible sample and try again. "))
    else                  # otherwise, KNN sample!
        df = knn_sample(bbf)
        eval!(bbf, df)
    end
    return
end

function globalsolve(md::GlobalModel)
    """ Creates and solves the global optimization model using the linear constraints from GlobalModel,
        and approximated nonlinear constraints from inside its BlackBoxFunctions."""
    clear_tree_constraints!(md); # remove trees from previous solve (if any).
    add_tree_constraints!(md); # refresh latest tree constraints.
    status = JuMP.optimize!(md.model);
    return status
end

function solution(md::GlobalModel)
    """ Returns the optimal solution of the solved global optimization model. """
    vals = getvalue.(md.vars)
    return DataFrame(Dict(var => vals[var] for var in gm.vars))
end

function evaluate_feasibility(md::GlobalModel)
    """ Evaluates each constraint at solution to make sure it is feasible. """
    soln = solution(md);
    feas = [];
    for fn in md.fns
        eval!(fn, soln)
    end
    fn_names = getfield.(md.fns, :name);
    infeas_idxs = fn_names[findall(vk -> md(vk).Y[end] .< 0, fn_names)]
    feas_idxs = fn_names[findall(vk -> md(vk).Y[end] .>= 0, fn_names)]
    return feas_idxs, infeas_idxs
end

function find_bounds!(gm::GlobalModel; all_bounds=true)
    """Finds the outer variable bounds of GlobalModel by solving over the linear constraints. """
    ubs = Dict(gm.vars .=> Inf)
    lbs = Dict(gm.vars .=> -Inf)
    # Finding bounds by min/maximizing each variable
    # TODO: REMOVE NONLINEAR CONSTRAINTS BEFORE LINEAR OPTIMIZATION.
    m = gm.model;
    x = gm.vars;
    current_bounds = get_bounds(gm);
    orig_objective = JuMP.objective_function(gm)
    @showprogress 0.5 "Finding bounds..." for var in gm.vars
        if isinf(current_bounds(var)) || all_bounds
            @objective(m, Min, var);
            JuMP.optimize!(m);
            lbs[var] = getvalue(var);
        end
        if isinf(current_bounds(var)) || all_bounds
            @objective(m, Max, var);
            JuMP.optimize!(m);
            ubs[var] = getvalue(var);
        end
    end
    # Revert objective
    @objective(m, Min, orig_objective)
    bounds = Dict(var => [lbs[var], ubs[var]] for var in x)
    bound!(gm, bounds)
    return
end

function import_trees(dir, gm::GlobalModel)
    """ Returns trees trained over given GlobalModel,
    where filename points to the model name. """
    trees = [IAI.read_json(string(dir, "_tree_", i, ".json")) for i=1:length(gm.fns)];
    return trees
end