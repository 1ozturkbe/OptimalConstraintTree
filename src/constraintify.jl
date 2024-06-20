""" 
    bounded_aux(x::Array{JuMP.VariableRef}, binary_var::JuMP.VariableRef)
    bounded_aux(x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, binary_var::JuMP.VariableRef)

Generates binary-bounded auxiliary variables and their bounding constraints of the same size as x + y.
"""
function bounded_aux(x::Array{JuMP.VariableRef}, binary_var::JuMP.VariableRef)
    var_bounds = get_bounds(x)
    aux_vars = @variable(x[1].model, [1:length(x)])
    bound_cons = []
    for i = 1:length(x) # Bounding auxiliary variables
        bds = var_bounds[x[i]]
        push!(bound_cons, @constraint(x[i].model, aux_vars[i] >= minimum(bds)*binary_var))
        push!(bound_cons, @constraint(x[i].model, aux_vars[i] <= maximum(bds)*binary_var))
    end
    aux_vars = (binary_var, aux_vars)
    return aux_vars, bound_cons
end

function bounded_aux(x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, binary_var::JuMP.VariableRef)
    aux_vars, bound_cons = bounded_aux(x, binary_var)
    aux_dep = @variable(y.model)
    push!(bound_cons, @constraint(y.model, aux_dep >= binary_var * JuMP.lower_bound(y)))
    push!(bound_cons, @constraint(y.model, aux_dep <= binary_var * JuMP.upper_bound(y)))
    aux_vars = (aux_vars[1], aux_vars[2], aux_dep)
    return aux_vars, bound_cons
end

"""
    $(TYPEDSIGNATURES)

Generates MI constraints from GlobalModel and its BlackBoxLearners(gm.bbls), and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))
    isempty(bbc.mi_constraints) || throw(OCTHaGOnException("BBC $(bbc.name) already has associated MI approximation."))
    isempty(bbc.leaf_variables) || throw(OCTHaGOnException("BBC $(bbc.name) already has associated MI variables."))
    if bbc.feas_ratio == 1.0 # Just a placeholder to show that the tree is "trained". 
        z_feas = @variable(gm.model, binary = true)
        bbc.mi_constraints = Dict(1 => [@constraint(gm.model, z_feas == 1)])
        bbc.leaf_variables = Dict(1 => (z_feas, []))
        if !isempty(bbc.lls)
            for ll in bbc.lls
                z_feas = @variable(gm.model, binary = true)
                ll.mi_constraints = Dict(1 => [@constraint(gm.model, z_feas == 1)])
                ll.leaf_variables = Dict(1 => (z_feas, []))
            end
        end
    elseif get_param(bbc, :reloaded)
        bbc.mi_constraints, bbc.leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx]; equality = bbc.equality, relax_var = bbc.relax_var, bigM = bbc.bigM)
        if !isempty(bbc.lls)
            for ll in bbc.lls
                ll.mi_constraints, ll.leaf_variables = add_feas_constraints!(gm.model, ll.vars, bbc.learners[idx]; equality = bbc.equality, relax_var = ll.relax_var, bigM = bbc.bigM, symbs = Symbol.(bbc.vars))
            end
        end
    elseif size(bbc.X, 1) == 0
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " has not been sampled yet, and is thus untrained."))
    elseif isempty(bbc.learners)
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " must be learned before tree constraints
                            can be generated."))
    elseif bbc.feas_ratio == 0.0
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " is INFEASIBLE but you tried to include it in
            your global problem. Find at least one feasible solution, train and try again."))
    elseif !get_param(gm, :ignore_accuracy) && !check_accuracy(bbc)
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " is inaccurately approximated. "))
    else
        bbc.mi_constraints, bbc.leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx]; equality = bbc.equality, relax_var = bbc.relax_var, bigM = bbc.bigM)
        if !isempty(bbc.lls)
            for ll in bbc.lls
                ll.mi_constraints, ll.leaf_variables = add_feas_constraints!(gm.model, ll.vars, bbc.learners[idx]; equality = bbc.equality, relax_var = ll.relax_var, bigM = bbc.bigM, symbs = Symbol.(bbc.vars))
            end
        end
    end
    bbc.active_leaves = []
    return
end

function add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    if size(bbr.X, 1) == 0 && !get_param(bbr, :reloaded)
        throw(OCTHaGOnException("Constraint " * string(bbr.name) * " has not been sampled yet, and is thus untrained."))
    elseif bbr.convex && !bbr.equality
        clear_tree_constraints!(gm, bbr)
        mi_constraints = Dict(1 => [])
        for i = Int64.(ceil.(size(bbr.X,1) .* rand(10)))
            push!(mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (bbr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
        end
        merge!(append!, bbr.mi_constraints, mi_constraints)
        if !isempty(bbr.lls)
            for lr in bbr.lls
                clear_tree_constraints!(gm, lr)
                mc = Dict(1 => [])
                # Number of initial cuts depends on the dimension of the constraint
                pt_idxs = Int64.(ceil.(size(bbr.X,1) .* rand(maximum([10, Int64(10*ceil(log(length(bbr.vars))))]))))
                update_gradients(bbr, pt_idxs)
                for i in pt_idxs
                    push!(mc[1], @constraint(gm.model, lr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (lr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
                end
                merge!(append!, lr.mi_constraints, mc)
                lr.active_leaves = [1]
            end
        end
        bbr.active_leaves = [1]
        return
    elseif isempty(bbr.learners)
        throw(OCTHaGOnException("Constraint " * string(bbr.name) * " must be learned before tree constraints
                            can be generated."))
    elseif isempty(bbr.ul_data)
        throw(OCTHaGOnException("Constraint " * string(bbr.name) * " is a Regressor, 
        but doesn't have a ORT and/or OCT with upper/lower bounding approximators!"))
    end
    if !isempty(bbr.lls)
        bbr.thresholds[idx].first == "reg" || 
            throw(OCTHaGOnException("BBRs with LinkedRegressors are only allowed ORT approximators."))
        length(bbr.active_trees) == 0 || 
            throw(OCTHaGOnException("BBRs with LinkedRegressors cannot have more than one active tree."))
    elseif bbr.thresholds[idx].first == "reg"
        (isempty(bbr.leaf_variables) || !isnothing(bbr.thresholds[idx].second)) ||
            throw(OCTHaGOnException("Please clear previous tree constraints from $(gm.name) " *
                "before adding an unthresholded regression constraints."))
        if !isnothing(bbr.thresholds[idx].second) && !isempty(bbr.active_trees)
            bbr.thresholds[active_upper_tree(bbr)].second == bbr.thresholds[idx].second || 
                throw(OCTHaGOnException("Upper-thresholded regressors must be preceeded by upper classifiers " * 
                                    "of the same threshold for $(bbr.name)."))
        end
    elseif bbr.thresholds[idx].first == "rfreg"
        isempty(bbr.leaf_variables) ||
            throw(OCTHaGOnException("Please clear previous tree constraints from $(gm.name) " *
            "before adding an random forest regressor constraints."))
            !isempty(bbr.lls) && throw(OCTHaGOnException("Random Forests cannot be used to approximate " *
                        "linked BBR $(bbr.name)."))
        isnothing(bbr.thresholds[idx].second) ||
            throw(OCTHaGOnException("RandomForestRegressors are not allowed upper bounds."))
        bbr.equality && 
            throw(OCTHaGOnException("RandomForest cannot approximate BBR $(bbr.name) since it is an equality constraint."))
    elseif bbr.thresholds[idx].first == "upper"
        all(collect(keys(bbr.mi_constraints)) .>= 0)  || throw(OCTHaGOnException("Please clear previous upper tree constraints from $(gm.name) " *
                                                          "before adding new constraints."))
    elseif bbr.thresholds[idx].first == "lower"
        all(collect(keys(bbr.mi_constraints)) .<= 0) || throw(OCTHaGOnException("Please clear previous lower tree constraints from $(gm.name) " *
                                                            "before adding new constraints."))
    end
    if bbr.thresholds[idx].first == "rfreg"
        isempty(bbr.lls) || throw(ErrorException("RandomForestRegressors augment LinkedLearners when MI approximations are made. To avoid bugs, BBRs approximated via RFs cannot have LinkedLearners."))
        trees = get_random_trees(bbr.learners[idx])
        bbr.mi_constraints, bbr.leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                    trees[1], bbr.ul_data[idx][1]; equality = bbr.equality, bigM = bbr.bigM, relax_var = bbr.relax_var,)
                    # mic = bbr.mi_constraints, lv = bbr.leaf_variables) 
        for i=2:length(trees)
            nll = LinkedRegressor(vars = bbr.vars, dependent_var = bbr.dependent_var)
            nll.mi_constraints, nll.leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                trees[i], bbr.ul_data[idx][i];
                                                equality = bbr.equality, bigM = bbr.bigM,)
                                                # mic = nll.mi_constraints,
                                                # lv = nll.leaf_variables)   
            push!(bbr.lls, nll)
        end
        bbr.active_trees[idx] = bbr.thresholds[idx]    
    else
       mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                bbr.learners[idx], bbr.ul_data[idx];
                                                equality = bbr.equality, bigM = bbr.bigM, relax_var = bbr.relax_var,)
                                                # mic = bbr.mi_constraints, 
                                                # lv = bbr.leaf_variables)
                                                
        merge!(append!, bbr.mi_constraints, mi_constraints)
        merge!(bbr.leaf_variables, leaf_variables)
        if !isempty(bbr.lls)
            for lr in bbr.lls
                lr_mi, lr_lv = add_regr_constraints!(gm.model, lr.vars, lr.dependent_var,
                                                bbr.learners[idx], bbr.ul_data[idx];
                                                equality = bbr.equality, bigM = bbr.bigM, relax_var = lr.relax_var,
                                                symbs = Symbol.(bbr.vars),)
                                                # mic = lr.mi_constraints,
                                                # lv = lr.leaf_variables)
                merge!(append!, lr.mi_constraints, lr_mi)
                merge!(lr.leaf_variables, lr_lv)
            end
        end
        if bbr.thresholds[idx].first == "upper"
            push!(bbr.mi_constraints[-1], @constraint(gm.model, bbr.dependent_var <= bbr.thresholds[idx].second))
            [push!(lr.mi_constraints[-1], @constraint(gm.model, lr.dependent_var <= bbr.thresholds[idx])) for lr in bbr.lls]
        elseif bbr.thresholds[idx].first == "lower"
            push!(bbr.mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= bbr.thresholds[idx].second))
            [push!(lr.mi_constraints[1], @constraint(gm.model, lr.dependent_var >= bbr.thresholds[idx])) for lr in bbr.lls]
        end
        bbr.active_trees[idx] = bbr.thresholds[idx]
    end
    bbr.active_leaves = []
    [lr.active_leaves = [] for lr in bbr.lls]
    return
end

function add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner})
    for bbl in bbls
        add_tree_constraints!(gm, bbl)
    end
    return
end

add_tree_constraints!(gm::GlobalModel) = add_tree_constraints!(gm, gm.bbls)

""" Computes the smallest possible big-M for a α'x ≥ threshold split in a tree. """
function compute_hyperplane_bigM_lower(threshold::Real, α::Vector, var_bounds::Vector)
    return threshold - sum(minimum(α[i] .* var_bounds[i]) for i=1:length(α))
end

""" Computes the smallest possible big-M for a α'x ≤ threshold split in a tree. """
function compute_hyperplane_bigM_upper(threshold::Real, α::Vector, var_bounds::Vector)
    return -threshold + sum(maximum(α[i] .* var_bounds[i]) for i=1:length(α))
end

""" Computes the smallest possible big-M for lower bounding regression in a tree leaf. """
function compute_regression_bigM_lower(β0::Real, β::Vector, var_bounds::Vector, y_bounds::Vector)
    return β0 + sum(maximum(β[i] .* var_bounds[i]) for i=1:length(β)) - minimum(y_bounds)
end

""" Computes the smallest possible big-M for upper bounding regression in a tree leaf. """
function compute_regression_bigM_upper(β0::Real, β::Vector, var_bounds::Vector, y_bounds::Vector)
    return -β0 - sum(minimum(β[i] .* var_bounds[i]) for i=1:length(β)) + maximum(y_bounds)
end

"""
Creates a set of binary feasibility constraints from a binary classification tree. 
Arguments:
- m: JuMP Model
- x: independent JuMP.Variables (features in lnr)
- lnr: A fitted OptimalTreeClassifier
- equality: whether the constraint is an equality. 
NOTE: mic and lv are only nonempty if we are adding an OCT approximation on top of an existing set of MI constraints and variables respectively. Leave defaults empty for basic usage. 
symbs helps add constraints to linked learners, since the varkeys in lnr are different to the variables in the linked learner. 
"""
function add_feas_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, lnr::IAI.OptimalTreeLearner;
                               equality::Bool = false, bigM::Bool = false,
                               relax_var::Union{Real, JuMP.VariableRef} = 0,
                               mic::Dict = Dict(), lv::Dict = Dict(), 
                               symbs = Symbol.(x))
    check_if_trained(lnr);
    all_leaves = find_leaves(lnr)
    # Add a binary variable for each leaf
    feas_leaves = []
    if lnr isa IAI.OptimalTreeClassifier
        feas_leaves = 
        [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
    else 
        feas_leaves = all_leaves
    end
    mi_constraints = Dict(leaf => [] for leaf in feas_leaves)
    mi_constraints[1] = []
    leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array}}()
    var_bounds = get_bounds(x)
    var_bounds = [var_bounds[var] for var in x] # ordered vector form
    if isempty(lv) # if the variables haven't already been created and passed!
        if bigM
            # big-M formulation variable generation and coupling
            for leaf in feas_leaves
                leaf_variables[leaf] = (@variable(m, binary = true), [])
            end
            push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in feas_leaves) == 1))
        else
            # big-M free formulation variable generation and coupling
            for leaf in feas_leaves
                leaf_variables[leaf], bd_mi = bounded_aux(x, @variable(m, binary=true))
                append!(mi_constraints[leaf], bd_mi)
            end
            push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in feas_leaves) == 1))
            [push!(mi_constraints[1], constr) 
                for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in feas_leaves) .== x)]
        end
    else
        leaf_variables = lv
    end
    # Getting lnr data
    upperDict, lowerDict = trust_region_data(lnr, symbs)
    for leaf in feas_leaves
        # ADDING TRUST REGIONS
        for (threshold, α) in upperDict[leaf] # upperDict is lower-bounded. 
            if bigM 
                M = compute_hyperplane_bigM_lower(threshold, α, var_bounds)
                push!(mi_constraints[leaf], @constraint(m, threshold ≤ sum(α .* x) + M * (1 - leaf_variables[leaf][1] + relax_var)))
            else
                push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] <= sum(α .* leaf_variables[leaf][2]) + relax_var))
            end
        end
        for (threshold, α) in lowerDict[leaf] # lowerDict is upper-bounded. 
            if bigM
                M = compute_hyperplane_bigM_upper(threshold, α, var_bounds)
                push!(mi_constraints[leaf], @constraint(m, M * (1 - leaf_variables[leaf][1] + relax_var) + threshold ≥ sum(α .* x)))
            else
                push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= sum(α .* leaf_variables[leaf][2])))
            end
        end
    end
    if equality
        infeas_leaves = [i for i in all_leaves if !Bool(IAI.get_classification_label(lnr, i))];
        [mi_constraints[leaf] = [] for leaf in infeas_leaves]     
        if isempty(lv)   
            if bigM
                # big-M formulation variable generation and coupling
                for leaf in infeas_leaves
                    leaf_variables[leaf] = (@variable(m, binary = true), [])
                end
                push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in infeas_leaves) == 1))  
            else
                for leaf in infeas_leaves
                    leaf_variables[leaf], bd_mi = bounded_aux(x, @variable(m, binary=true))
                    append!(mi_constraints[leaf], bd_mi)
                end
                push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in infeas_leaves) == 1))
                [push!(mi_constraints[1], constr) 
                    for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in infeas_leaves) .== x)]
            end
        end
        for leaf in infeas_leaves
            # ADDING TRUST REGIONS
            for (threshold, α) in upperDict[leaf] # upperDict is lower-bounded. 
                if bigM 
                    M = compute_hyperplane_bigM_lower(threshold, α, var_bounds)
                    push!(mi_constraints[leaf], @constraint(m, threshold ≤ sum(α .* x) + M * (1 - leaf_variables[leaf][1] + relax_var)))
                else
                    push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] <= sum(α .* leaf_variables[leaf][2]) + relax_var))
                end
            end
            for (threshold, α) in lowerDict[leaf] # lowerDict is upper-bounded. 
                if bigM
                    M = compute_hyperplane_bigM_upper(threshold, α, var_bounds)
                    push!(mi_constraints[leaf], @constraint(m, threshold + M * (1 - leaf_variables[leaf][1] + relax_var) >= sum(α .* x)))
                else
                    push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= sum(α .* leaf_variables[leaf][2])))
                end
            end
        end
    end
    merge!(append!, mi_constraints, mic)
    return mi_constraints, leaf_variables
end

"""
    $(TYPEDSIGNATURES)

Creates a set of MIO constraints from a OptimalTreeClassifier that thresholds a BlackBoxRegressor.

Arguments:
- m: JuMP.Model
- x:: independent JuMP.Variables (features of learner)
- y:: dependent JuMP.Variable
- lnr:: A fitted OptimalTreeLearner
- ul_data:: Upper and lower bounding hyperplanes for data in leaves of lnr (empty by default)
- symbs:: The varkeys corresponding to the lnr.
"""
function add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, 
                               lnr::IAI.OptimalTreeLearner,
                               ul_data::Dict; equality::Bool = false, bigM::Bool = false,
                               relax_var::Union{Real, JuMP.VariableRef} = 0, symbs = Symbol.(x))
    # TODO: Determine whether I should relax approximator or trust regions or both. 
    if lnr isa OptimalTreeRegressor                
        check_if_trained(lnr)
        all_leaves = find_leaves(lnr)
        # Add a binary variable for each leaf
        mi_constraints = Dict(leaf => [] for leaf in all_leaves)
        mi_constraints[1] = []
        leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}()
        var_bounds = get_bounds(x)
        var_bounds = [var_bounds[var] for var in x]
        y_bounds = get_bound(y)
        if bigM
            for leaf in all_leaves
                leaf_variables[leaf] = (@variable(m, binary = true), [], @variable(m)) # third item in tuple is placeholder. TODO: find a more elegant bookkeeping method. 
            end
            push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in all_leaves) == 1))
        else
            for leaf in all_leaves
                leaf_variables[leaf], bd_mi = bounded_aux(x, y, @variable(m, binary=true))
                append!(mi_constraints[leaf], bd_mi)
            end
            push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in all_leaves) == 1))
            [push!(mi_constraints[1], constr) 
                for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in all_leaves) .== x)]
            push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][3] for leaf in all_leaves) == y))    
        end
        # Getting lnr data
        pwlDict = pwl_constraint_data(lnr, symbs)
        upperDict, lowerDict = trust_region_data(lnr, symbs)
        for leaf in all_leaves
            # ADDING PWL APPROXIMATIONS
            β0, β = pwlDict[leaf]
            if bigM
                if equality
                    M_upper = compute_regression_bigM_upper(β0, β, var_bounds, y_bounds)
                    M_lower = compute_regression_bigM_lower(β0, β, var_bounds, y_bounds)
                    # Use the ridge regressor!
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* x) + β0 + M_upper * (1 - leaf_variables[leaf][1] + relax_var) >= y))
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M_lower * (1 - leaf_variables[leaf][1] + relax_var)))
                elseif !isempty(ul_data) 
                    # Use only the lower approximator
                    β0, β = ul_data[leaf] # update the lower approximator
                    M_lower = compute_regression_bigM_lower(β0, β, var_bounds, y_bounds)
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M_lower * (1 - leaf_variables[leaf][1] + relax_var)))
                else 
                    # Use only the ridge regressor as the lower approximator. 
                    M_lower = compute_regression_bigM_lower(β0, β, var_bounds, y_bounds)
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M_lower * (1 - leaf_variables[leaf][1] + relax_var)))
                end
            else
                if equality
                    # Use the ridge regressor!
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + β0 * leaf_variables[leaf][1] + relax_var >= leaf_variables[leaf][3]))
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + β0 * leaf_variables[leaf][1] <= leaf_variables[leaf][3] + relax_var))
                elseif !isempty(ul_data) 
                    # Use only the lower approximator
                    β0, β = ul_data[leaf] # update the lower approximator
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + β0 * leaf_variables[leaf][1] <= leaf_variables[leaf][3] + relax_var))
                else 
                    # Use only the ridge regressor as the lower approximator. 
                    push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + β0 * leaf_variables[leaf][1] <= leaf_variables[leaf][3] + relax_var))
                end
            end
            # ADDING TRUST REGIONS
            for (threshold, α) in upperDict[leaf] # upperDict is lower-bounded. 
                if bigM 
                    M = compute_hyperplane_bigM_lower(threshold, α, var_bounds)
                    push!(mi_constraints[leaf], @constraint(m, threshold ≤ sum(α .* x) + M * (1 - leaf_variables[leaf][1] + relax_var)))
                else
                    push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] <= sum(α .* leaf_variables[leaf][2]) + relax_var))
                end
            end
            for (threshold, α) in lowerDict[leaf] # lowerDict is upper-bounded. 
                if bigM
                    M = compute_hyperplane_bigM_upper(threshold, α, var_bounds)
                    push!(mi_constraints[leaf], @constraint(m, M * (1 - leaf_variables[leaf][1] + relax_var) + threshold ≥ sum(α .* x)))
                else
                    push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= 
                    sum(α .* leaf_variables[leaf][2])))
                end
            end
        end
        return mi_constraints, leaf_variables
    elseif lnr isa OptimalTreeClassifier
        # Add a binary variable for each leaf
        all_leaves = find_leaves(lnr)
        # Add a binary variable for each leaf
        feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
        mi_constraints = Dict(leaf => [] for leaf in feas_leaves)
        mi_constraints[1] = []
        leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}()
        for leaf in feas_leaves
            leaf_variables[leaf], bd_mi = bounded_aux(x, y, @variable(m, binary=true))
            append!(mi_constraints[leaf], bd_mi)
        end
        push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in feas_leaves) == 1))
        [push!(mi_constraints[1], constr) 
            for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in feas_leaves) .== x)]
        push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][3] for leaf in feas_leaves) == y))
        mi_constraints, leaf_variables = add_feas_constraints!(m, x, lnr, equality = equality, bigM = bigM, mic = mi_constraints, lv = leaf_variables)
        if !isempty(ul_data)
            if all(keys(ul_data) .<= 0) || !all(keys(ul_data) .>= 0) # means an upper or upperlower bounding tree
                mi_constraints = Dict(-key => value for (key, value) in mi_constraints) # hacky sign flipping for upkeep. 
                leaf_variables = Dict(-key => value for (key, value) in leaf_variables) 
            end
            for leaf in collect(keys(ul_data))
                γ0, γ = ul_data[leaf]
                if !haskey(mi_constraints, leaf) # occurs with ORTS with bounding hyperplanes
                    mi_constraints[leaf] = [@constraint(m, leaf_variables[-leaf][3] <= γ0 * leaf_variables[-leaf][1] + 
                        sum(γ .* leaf_variables[-leaf][2]) + relax_var)]
                elseif leaf <= 0
                    push!(mi_constraints[leaf], @constraint(m, leaf_variables[leaf][3] <= γ0 * leaf_variables[leaf][1] + 
                        sum(γ .* leaf_variables[leaf][2]) + relax_var))
                else
                    push!(mi_constraints[leaf], @constraint(m, leaf_variables[leaf][3] + relax_var >= 
                    γ0 * leaf_variables[leaf][1] + sum(γ .* leaf_variables[leaf][2])))
                end
            end
        end
        return mi_constraints, leaf_variables
    end
end

""" Clears upper-bounding constraints from a BBR and its associated GM. """
function clear_upper_constraints!(gm, bbr::Union{BlackBoxRegressor, LinkedRegressor})
    for (leaf_key, leaf_constrs) in bbr.mi_constraints
        if leaf_key <= 0
            while !isempty(leaf_constrs)
                constr = pop!(leaf_constrs)
                if is_valid(gm.model, constr)
                    delete(gm.model, constr)
                else
                    push!(leaf_constrs, constr) # make sure to put the constraint back. 
                    throw(OCTHaGOnException("Bug: Constraints could not be removed."))
                end
            end
            delete!(bbr.mi_constraints, leaf_key)
        end
    end
    for (leaf_key, (bin_var, leaf_vars, aux_dep)) in bbr.leaf_variables
        if leaf_key <= 0
            for leaf_var in leaf_vars
                if is_valid(gm.model, leaf_var)
                    delete(gm.model, leaf_var)
                else
                    throw(OCTHaGOnException("Bug: Variables could not be removed."))
                end
            end
            if is_valid(gm.model, bin_var)
                delete(gm.model, bin_var)
            else
                throw(OCTHaGOnException("Bug: Binary variable could not be removed."))
            end
            if is_valid(gm.model, aux_dep)
                delete(gm.model, aux_dep)
            else
                throw(OCTHaGOnException("Bug: Aux dependent variable could not be removed."))
            end
            delete!(bbr.leaf_variables, leaf_key)
        end
    end
    if bbr isa BlackBoxRegressor
        idx = active_upper_tree(bbr)
        delete!(bbr.active_trees, idx)
        for lr in bbr.lls
            clear_upper_constraints!(gm, lr)
        end
    end
    return
end

""" Clears lower-bounding constraints from a BBR and its associated GM. """
function clear_lower_constraints!(gm, bbr::Union{BlackBoxRegressor, LinkedRegressor})
    for (leaf_key, leaf_constrs) in bbr.mi_constraints
        if leaf_key >= 0
            while !isempty(leaf_constrs)
                constr = pop!(leaf_constrs)
                if is_valid(gm.model, constr)
                    delete(gm.model, constr)
                else
                    push!(leaf_constrs, constr) # make sure to put the constraint back. 
                    throw(OCTHaGOnException("Bug: Constraints could not be removed."))
                end
            end
            delete!(bbr.mi_constraints, leaf_key)
        end
    end
    for (leaf_key, (bin_var, leaf_vars, aux_dep)) in bbr.leaf_variables
        if leaf_key >= 0
            for leaf_var in leaf_vars
                if is_valid(gm.model, leaf_var)
                    delete(gm.model, leaf_var)
                else
                    throw(OCTHaGOnException("Bug: Variables could not be removed."))
                end
            end
            if is_valid(gm.model, bin_var)
                delete(gm.model, bin_var)
            else
                throw(OCTHaGOnException("Bug: Binary variable could not be removed."))
            end
            if is_valid(gm.model, aux_dep)
                delete(gm.model, aux_dep)
            else
                throw(OCTHaGOnException("Bug: Aux dependent variable could not be removed."))
            end
            delete!(bbr.leaf_variables, leaf_key)
        end
    end
    bbr.active_leaves = []
    if bbr isa BlackBoxRegressor
        idx = active_lower_tree(bbr)
        delete!(bbr.active_trees, idx)
        for lr in bbr.lls
            clear_lower_constraints!(gm, lr)
        end
    end
    return
end

""" 
    $(TYPEDSIGNATURES)

Clears the MI-approximating constraints and variables in GlobalModel and its sub-structs.
"""
function clear_tree_constraints!(gm::GlobalModel, bbc::Union{BlackBoxClassifier, LinkedClassifier})
    for (leaf_key, leaf_constrs) in bbc.mi_constraints
        while !isempty(leaf_constrs)
            constr = pop!(leaf_constrs)
            if is_valid(gm.model, constr)
                delete(gm.model, constr)
            else
                push!(leaf_constrs, constr) # make sure to put the constraint back. 
                throw(OCTHaGOnException("Bug: Constraints could not be removed."))
            end
        end
        delete!(bbc.mi_constraints, leaf_key)
    end
    for (leaf_key, (bin_var, leaf_vars)) in bbc.leaf_variables
        for leaf_var in leaf_vars
            if is_valid(gm.model, leaf_var)
                delete(gm.model, leaf_var)
            else
                throw(OCTHaGOnException("Bug: Variables could not be removed."))
            end
        end
        if is_valid(gm.model, bin_var)
            delete(gm.model, bin_var)
        else
            throw(OCTHaGOnException("Bug: Binary variable could not be removed."))
        end
        delete!(bbc.leaf_variables, leaf_key)
    end
    bbc.active_leaves = []
    if bbc isa BlackBoxClassifier
        for lc in bbc.lls
            clear_tree_constraints!(gm, lc)
        end
    end
    return
end

function clear_tree_constraints!(gm::GlobalModel, bbr::Union{BlackBoxRegressor, LinkedRegressor})
    clear_lower_constraints!(gm, bbr)
    clear_upper_constraints!(gm, bbr)
    return 
end

function clear_tree_constraints!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    for bbl in bbls
        clear_tree_constraints!(gm, bbl)
    end
    return
end

clear_tree_constraints!(gm::GlobalModel) = clear_tree_constraints!(gm, gm.bbls)

"""
    update_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    update_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))

Updates the MI constraints associated with a BBL. 
For BBRs, makes sure to replace the appropriate lower/upper/regressor approximations. 
"""
function update_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    if isempty(bbr.active_trees) # no mi constraints yet. 
        add_tree_constraints!(gm, bbr, idx)
    elseif length(bbr.active_trees) == 1 # one set of mi_constraints
        new_threshold = bbr.thresholds[idx]
        active_idx = collect(keys(bbr.active_trees))[1]
        last_threshold = bbr.active_trees[active_idx]
        if new_threshold.first == last_threshold.first
            clear_tree_constraints!(gm, bbr)
            add_tree_constraints!(gm, bbr, idx)
            return
        elseif new_threshold == Pair("reg", nothing) || last_threshold == Pair("reg", nothing) || 
            last_threshold.first == "upperlower" || new_threshold.first == "upperlower"
            clear_tree_constraints!(gm, bbr)
            add_tree_constraints!(gm, bbr, idx)
            return
        elseif new_threshold.first == "reg" && last_threshold.first == "upper" || 
                last_threshold.first == "reg" && new_threshold.first == "upper"
            if new_threshold.second == last_threshold.second # if corresponding upper thresholds. 
                add_tree_constraints!(gm, bbr, idx)
                return
            else
                clear_tree_constraints!(gm, bbr)
                add_tree_constraints!(gm, bbr, idx)
                return
            end
        elseif last_threshold.first in valid_uppers && new_threshold.first in valid_lowers || 
            last_threshold.first in valid_lowers && new_threshold.first in valid_uppers
            add_tree_constraints!(gm, bbr, idx)
            return
        else # updating a lower or upper threshold
            clear_tree_constraints!(gm, bbr)
            add_tree_constraints!(gm, bbr, idx)
            return
        end
    elseif length(bbr.active_trees) == 2 # two sets of mi_constraints
        if bbr.thresholds[idx] == Pair("reg", nothing) || bbr.thresholds[idx].first == "upperlower"
            clear_tree_constraints!(gm, bbr)                                                  
            add_tree_constraints!(gm, bbr, idx)                                               
            return
        end
        hypertype = bbr.thresholds[idx].first # otherwise, check approximation type
        if hypertype in valid_lowers
            clear_lower_constraints!(gm, bbr)
        elseif hypertype in valid_uppers
            clear_upper_constraints!(gm, bbr)
        else
            throw(OCTHaGOnException("Hypertype $(hypertype) not recognized."))
        end
        add_tree_constraints!(gm, bbr, idx)
        return 
    else
        throw(OCTHaGOnException("$(gm.name) has too many active lower/upper bounding regressors."))
    end
end

function update_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))
    clear_tree_constraints!(gm, bbc)
    add_tree_constraints!(gm, bbc, idx)
    return
end