using JLBoost
using LossFunctions: L2DistLoss


@with_kw mutable struct GBM_Classifier <: AbstractClassifier 
    # Arguments
    max_depth::Int64 = 4
    nrounds::Int64 = 3
    solver = CPLEX_SILENT
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing

    # Model data
    gbm::Union{Nothing, JLBoostTreeModel} = nothing
    use_epsilon::Bool = false # Whether or not the equality is approximated with epsilon tolerance
    equality::Bool = false # Whether or not we are dealing with an equality constraint
    thres::Real = 0.5 # Classification threshold

end

@with_kw mutable struct GBM_Regressor <: AbstractRegressor
    # Arguments
    max_depth::Int64 = 6
    nrounds::Int64 = 2
    solver = CPLEX_SILENT
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing
    
    # Model data
    gbm::Union{Nothing, JLBoostTreeModel} = nothing
    equality::Bool = false # Whether or not we are dealing with an equality constraint
end




function find_all_leaves(tree::AbstractJLBoostTree)
    if length(tree.children) == 0
        return [tree]
    else
        leaves = []
        for child in tree.children
            append!(leaves, find_all_leaves(child))
        end
        return leaves
    end
end

function update_leaf_df!(df::DataFrame, leaf::AbstractJLBoostTree, cols::Vector{String}, leaf_id::Int64, leaf_weight::Float64)
    
    if isnothing(leaf.parent)
        return
    end
    
    # Initialize all variables to 0
    d = Dict(c => 0.0 for c in cols)
    
    multiplier = (leaf == leaf.parent.children[1] ? 1 : -1)

    # Change value of splitting feature
    d[String(leaf.parent.splitfeature)] = multiplier
    
    #Determine threshold    
    d["threshold"] = multiplier*leaf.parent.split
    
    # Keep track of the leaf id and the leaf prediction
    d["leaf_id"] = leaf_id
    d["prediction"] = leaf_weight
    
    df_new = DataFrame(d)
    
    append!(df, df_new)
    update_leaf_df!(df, leaf.parent, cols, leaf_id, leaf_weight)
end

function find_leaf_df(tree::AbstractJLBoostTree, bbl::BlackBoxLearner)

    df = DataFrame()
    leaves = find_all_leaves(tree);
    #leaf = leaves[1];
    cols = names(bbl.X);

    i = 1
    for leaf in leaves 
        update_leaf_df!(df, leaf, cols, i, leaf.weight)
        i += 1
    end
    
    return df
end

function embed_single_tree(gm::GlobalModel, bbl::BlackBoxLearner, tree_id::Int64, tree::AbstractJLBoostTree; M=100000, ro_factor=0)
    m = gm.model;
    cols = names(bbl.X)
    x = bbl.vars;

    # Calculate the df that describes the constraints
    df = find_leaf_df(tree, bbl)
    
    # Predictions for each leaf
    leaf_predictions = combine(first, groupby(df, :leaf_id))[!, "prediction"];

    # Splitting thresholds
    intercept = df[!, "threshold"];

    # Splitting coefficients (i.e. which variable is active)
    coeffs = Matrix(df[!, cols]);

    # Leaf ids
    l_ids = convert.(Int64, df[!, "leaf_id"]);
    n_leaves = length(unique(l_ids))

    # Create 1 variable for every leaf
    var_name = eval(Meta.parse(":d$(tree_id)"));
    m[var_name] = @variable(m, [i=1:n_leaves], Bin, base_name=string(var_name));
    leaf_vars = m[var_name];

    # Create 1 outcome variable for the whole tree
    var_name = eval(Meta.parse(":y$(tree_id)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    outcome_var = m[var_name]

    # If the coefficient is -1, force strict inequality
    strict_ineq_epsilons = (1e-6)*(sum(coeffs, dims=2) .== -1)

    constrs = []

    push!(constrs, @constraint(m, outcome_var == leaf_predictions'*leaf_vars));
    push!(constrs, @constraint(m, sum(leaf_vars[i] for i=1:n_leaves) == 1));
    # .+ strict_ineq_epsilons
    if ro_factor == 0
        append!(constrs, @constraint(m, coeffs*x .<= intercept.+M*(1 .- leaf_vars[l_ids])));
    else
        for i in 1:size(coeffs, 1)

            # Robust coefficient matrix
            P = ro_factor*diagm(1.0*coeffs[i, :])
            
            # Create variables that will be used for robustness
            var_name = eval(Meta.parse(":t_rnn_$(bbl.name)_$(tree_id)_$(i)"));
            m[var_name] = @variable(m, base_name=string(var_name));
            t_var = m[var_name];

            push!(constrs, @constraint(m, sum(coeffs[i, :].*x) + strict_ineq_epsilons[i] + t_var <= intercept[i] + M*(1 - leaf_vars[l_ids][i])));
            append!(constrs, @constraint(m, P*x .<= t_var))
            append!(constrs, @constraint(m, -P*x .<= t_var))
        end
    end
    
    return constrs, outcome_var;
end

function gbm_embed_helper(lnr::Union{GBM_Regressor, GBM_Classifier}, gm::GlobalModel, bbl::Union{BlackBoxClassifier, BlackBoxRegressor}, lb=-Inf, ub=Inf; kwargs...) 
    
    trees = lnr.gbm.jlt;
    
    m = gm.model;
    
    all_constraints = []
    outcome_vars = []
    etas = []
    for (i, tree) in enumerate(trees)
        constrs, tree_outcome = embed_single_tree(gm, bbl, i, tree; M=100000, ro_factor=get_param(gm, :ro_factor));
        push!(outcome_vars, tree_outcome)
        push!(etas, tree.eta)
        append!(all_constraints, constrs)
    end

    # Define final outcome variable
    var_name = eval(Meta.parse(":$(bbl.name)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    final_outcome = m[var_name];
    
    if !isnothing(lnr.dependent_var)
        push!(all_constraints, @constraint(m, final_outcome == lnr.dependent_var))
    else
        relax_var = gm.relax_coeff ==0 ? 0 : gm.relax_var;
        if lb != -Inf
            push!(all_constraints, @constraint(m, final_outcome >= lb-relax_var));
        end
        if ub != Inf
            push!(all_constraints, @constraint(m, final_outcome <= ub+relax_var));
        end
    end
    # Final outcome variable is the weighted average
    # of the sub-trees
    push!(all_constraints, @constraint(m, outcome_vars'*etas./sum(etas) == final_outcome));
    

    return Dict(1 => all_constraints), Dict()
end


function convert_to_binary(lnr::GBM_Classifier, Y::Array)
    return (lnr.equality && lnr.use_epsilon ? 1*(abs.(Y .- lnr.thres) .<= EPSILON) : 1*(Y .>= lnr.thres));
end


function fit_cls_helper(lnr::GBM_Classifier, X::DataFrame, Y::Array; equality=false)
    lnr.equality = equality

    df = deepcopy(X)

    Y_hat = 1*(Y .>= 0) 
    
    if equality
        tmp = abs.(Y) .<= EPSILON
        positive_sample_fraction = sum(tmp)/length(Y);
        if positive_sample_fraction >= 0.1
            Y_hat = tmp
            lnr.use_epsilon = true
        else 
            # @error("Not enough samples to GBM approximate equality constraint: $(positive_sample_fraction)")
            # In this case, we will continue modeling as inequality instead of equality
            println("Not enough samples to GBM approximate equality constraint: $(positive_sample_fraction)")
        end
    end
    df[!,"output"] = Y_hat;
    return jlboost(df, "output"; verbose=false, max_depth = lnr.max_depth, nrounds=lnr.nrounds);
end

"""
Fit gbm in classification task
"""
function fit!(lnr::GBM_Classifier, X::DataFrame, Y::Array; equality=false)
    lnr.gbm = fit_cls_helper(lnr, X, Y; equality=equality)
end

"""
Fit gbm in regression task
"""
function fit!(lnr::GBM_Regressor , X::DataFrame, Y::Array; equality=false)
    
    lnr.equality = equality

    df = deepcopy(X)
    df[!,"output"] = Y;

   
	lnr.gbm = jlboost(df, "output", setdiff(Tables.columnnames(df), ["output"]), fill(0.0, nrow(df)), L2DistLoss(), max_depth = lnr.max_depth, nrounds=lnr.nrounds)

    #lnr.gbm = jlboost(df, "output"; verbose=false, max_depth = lnr.max_depth);
end

"""
Predict using gbm in classification task
"""
function predict(lnr::GBM_Classifier, X::DataFrame; continuous=false)
    
    if isnothing(lnr.gbm)
        error("GBM model hasn't been fitted yet")
    end
    y = JLBoost.predict(lnr.gbm, X)
    if !continuous
        y = convert_to_binary(lnr, y)
    end
    
    return y
end


"""
Predict using gbm in regression task
"""
function predict(lnr::GBM_Regressor, X::DataFrame)
    
    if isnothing(lnr.gbm)
        error("GBM model hasn't been fitted yet")
    end
    y = JLBoost.predict(lnr.gbm, X)
    return y
end


"""
Evaluate using gbm in classification task
"""
function evaluate(lnr::GBM_Classifier, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = classification_evaluation()

    score = evaluator.second(y_pred, convert_to_binary(lnr, Y))
    return score
end

"""
Evaluate using gbm in regression task
"""
function evaluate(lnr::GBM_Regressor, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = regression_evaluation()

    score = evaluator.second(y_pred, Y)
    return score
end


"""
Embed MIO constraints on GBM classifier
"""
function embed_mio!(lnr::GBM_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)

    return gbm_embed_helper(lnr, gm, bbl, lnr.thres)
end


"""
Embed MIO constraints on GBM regressor
"""
function embed_mio!(lnr::GBM_Regressor, gm::GlobalModel, bbl::BlackBoxRegressor; kwargs...)
   
    if lnr.equality
        return gbm_embed_helper(lnr, gm, bbl, -EPSILON, EPSILON)
    else 
        return gbm_embed_helper(lnr, gm, bbl, 0)
    end
end


"""
Test GBM 
"""

# function test_gbm()

#     X = rand(100, 50)
    
#     y = 1.0*(sum(X', dims=[1]).>50)[1,:]#

#     lnr = GBM_Classifier()

#     lnr.fit!(lnr, X, y);
    
#     y_hat = lnr.predict(lnr, X)
#     accuracy = 100*sum(y .== y_hat)/length(y)
#     println("Accuracy $(accuracy)")
    
#     y_hat = lnr.predict(lnr, X; continuous=true)
#     accuracy = 100*sum(y .== y_hat)/length(y)
#     println("Accuracy $(accuracy)")
    
#     return y, y_hat
# end