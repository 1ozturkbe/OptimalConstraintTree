

""" Helper function for merging learner arguments. """
function merge_kwargs(valid_keys; kwargs...)
    nkwargs = Dict{Symbol, Any}() 
    for item in kwargs
        if item.first in valid_keys
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! 
TODO: Add nkwargs that can be changed! """
function fit_regressor_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}()
    for item in kwargs
        if item.first in keys(nkwargs)
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! 
TODO: Add nkwargs that can be changed! """
function fit_classifier_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}(:sample_weight => :autobalance)
    for item in kwargs
        if item.first in keys(nkwargs)
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Function that preprocesses merging of kwargs for IAI.OptimalTreeClassifier, to avoid errors! """
function classifier_kwargs(; kwargs...)
    valid_keys = [:random_seed, :max_depth, :cp, :minbucket, 
                  :fast_num_support_restarts, :localsearch, 
                  :ls_num_hyper_restarts, :ls_num_tree_restarts, 
                  :hyperplane_config, :criterion]
    merge_kwargs(valid_keys; kwargs...)
end

""" Function that preprocesses merging of kwargs for IAI.OptimalTreeRegressor, to avoid errors! """
function regressor_kwargs(; kwargs...)
    valid_keys = [:random_seed, :max_depth, :cp, :minbucket, 
                  :fast_num_support_restarts, :localsearch, 
                  :ls_num_hyper_restarts, :ls_num_tree_restarts, 
                  :hyperplane_config,
                  :regression_sparsity, :regression_weighted_betas,
                  :regression_lambda, :criterion]
    return merge_kwargs(valid_keys; kwargs...)
end

""" 
    learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::IAI.OptimalTreeLearner, 
                          idxs::Union{Nothing, Array}=nothing; kwargs...)

Wrapper around IAI.fit! for constraint learning.
Arguments:
    lnr: OptimalTreeClassifier or OptimalTreeRegressor
    X: matrix of feature data
    Y: matrix of constraint data.
Returns:
    lnr: Fitted Learner corresponding to the data
NOTE: kwargs get unpacked here, all the way from learn_constraint!.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::Union{IAI.OptimalTreeLearner, 
                                                                     IAI.Heuristics.RandomForestLearner}, 
                          idxs::Union{Nothing, Array}=nothing;use_test_set=true, kwargs...)
    n_samples, n_features = size(X);
    @assert n_samples == length(Y);
    # Making sure that we only consider relevant features.
    if !isnothing(idxs)
        IAI.set_params!(lnr, split_features = idxs)
        if typeof(lnr) in [IAI.OptimalTreeRegressor, IAI.Heuristics.RandomForestRegressor]
            IAI.set_params!(lnr, regression_features = idxs)
        end
    else
        IAI.set_params!(lnr, split_features = All())
        if typeof(lnr) in [IAI.OptimalTreeRegressor, IAI.Heuristics.RandomForestRegressor]
            IAI.set_params!(lnr, regression_features = All())
        end
    end

    if use_test_set 
        (X_train, y_train), (X_test, y_test) = train_test_split(X, Y; at = 0.67)
    else 
        X_train, y_train = X, Y  
        X_test, y_test = X, Y
    end 
    
    IAI.fit!(lnr, X_train, y_train; kwargs...)
    #score = IAI.score(lnr, X_test, y_test)

    evaluator = lnr isa IAI.OptimalTreeRegressor ? regression_evaluation() : classification_evaluation();
    y_pred = IAI.predict(lnr, X_test)

    score = evaluator.second(y_pred, y_test)

    return lnr, score
end


""" 
    learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::AbstractRegressor, idxs::Union{Nothing, Array}=nothing; 
                        use_test_set=true, kwargs...)

Wrapper around fit! of the different regressors (e.g. CART, GBM)
Arguments:
    lnr: AbstractRegressor
    X: matrix of feature data
    Y: matrix of constraint data.
Returns:
    lnr: Fitted Learner corresponding to the data
NOTE: kwargs get unpacked here, all the way from learn_constraint!.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::AbstractRegressor, idxs::Union{Nothing, Array}=nothing; 
    use_test_set=true, equality=false, kwargs...)

    if !isnothing(idxs)
        println("Feature splitting not suppoerted yet: using all features")
    end

    if use_test_set 
        (X_train, y_train), (X_test, y_test) = train_test_split(X, Y; at = 0.67)
    else 
        X_train, y_train = X, Y  
        X_test, y_test = X, Y
    end 

    fit!(lnr, X_train, y_train; equality = equality)

    # y_pred = lnr.predict(lnr, X_test)

    # score = r2_score(y_pred, y_test)

    score = evaluate(lnr, X_test, y_test)

    return lnr, score
end


""" 
    learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::AbstractClassifier, idxs::Union{Nothing, Array}=nothing; 
                        use_test_set=true, kwargs...)

Wrapper around fit! of the different regressors (e.g. CART, GBM)
Arguments:
    lnr: AbstractClassifier
    X: matrix of feature data
    Y: matrix of constraint data (CONTINUOUS, not binary)
Returns:
    lnr: Fitted Learner corresponding to the data
NOTE: kwargs get unpacked here, all the way from learn_constraint!.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::AbstractClassifier, idxs::Union{Nothing, Array}=nothing; 
    use_test_set=true, equality=false, kwargs...)

    if !isnothing(idxs)
        println("Feature splitting not supported yet: using all features")
    end

    if use_test_set 
        (X_train, y_train), (X_test, y_test) = train_test_split(X, Y; at = 0.67)
    else 
        X_train, y_train = X, Y  
        X_test, y_test = X, Y
    end 
    
    fit!(lnr, X_train, y_train; equality = equality)

    #y_test = equality ? 1*(abs.(y_test).<= EPSILON) : 1*(y_test .>= 0)
    
    #y_pred = lnr.predict(lnr, X_test)

    score = evaluate(lnr, X_test, y_test)

    return lnr, score
end

function learn_from_data_ro!(X::DataFrame, Y::AbstractArray, lnr::Union{IAI.OptimalTreeLearner, 
                                                                     IAI.Heuristics.RandomForestLearner}, bbl::BlackBoxClassifier,
                          idxs::Union{Nothing, Array}=nothing;use_test_set=true, kwargs...)
    
    
    if lnr isa IAI.OptimalTreeClassifier
        all_idx = collect(1:size(X, 1))
        upper_dicts = []
        lower_dicts = []
        
        # for i= 1:1
        #     try
        #         println("Robust exploration $(i)")
        #         sub_idx = StatsBase.sample(all_idx, trunc(Int, size(X,1)*0.8))
        #         lnr, score = learn_from_data!(copy(X[sub_idx,:]), copy(Y[sub_idx]), lnr, idxs; use_test_set=use_test_set, kwargs...)
        #         upper_dict, lower_dict = trust_region_data(lnr, Symbol.(bbl.expr_vars))
                
                
        #         push!(upper_dicts, upper_dict)
        #         push!(lower_dicts, lower_dict)
        #     catch
        #         println("Robust exploration $(i) failed")
        #         continue
        #     end
        # end
        
        A_all_u, t_all_u = construct_tr_matrixes(upper_dicts; upper_dict=true)
        A_all_l, t_all_l = construct_tr_matrixes(lower_dicts; upper_dict=false)

        lnr, score = learn_from_data!(copy(X), copy(Y), lnr, idxs; use_test_set=use_test_set, kwargs...)

        return lnr, score, (A_all_u, t_all_u, A_all_l, t_all_l)
    end
    
    return lnr, score
end

# function fit_and_evaluate!(lnr::AbstractClassifier, bbc::BlackBoxClassifier; equality=false)

#     EPSILON = 1e-1

#     (X_train, y_train), (X_test, y_test) = train_test_split(bbc.X, bbc.Y; at = 0.67)

#     lnr.fit!(lnr, X_train, y_train; equality = equality)

#     y_pred = lnr.predict(lnr, X_test)

#     y_test = equality ? 1*(abs.(y_test).<= EPSILON) : 1*(y_test .>= 0);

#     score = roc_auc_score(y_test, y_pred)

#     return score
# end

# function fit_and_evaluate!(lnr::AbstractRegressor, bbr::BlackBoxRegressor; equality=false)

#     EPSILON = 1e-1

#     (X_train, y_train), (X_test, y_test) = train_test_split(bbr.X, bbr.Y; at = 0.67)

#     lnr.fit!(lnr, X_train, y_train; equality = equality)

#     y_pred = lnr.predict(lnr, X_test)

#     score = r2_score(y_pred, y_true)

#     return score
# end

""" Checks that a BlackBoxClassifier has enough feasible/infeasible samples. """
function check_feasibility(bbc::BlackBoxClassifier)
    return bbc.feas_ratio >= get_param(bbc, :threshold_feasibility)
end

check_feasibility(bbr::BlackBoxRegressor) = true

function check_feasibility(gm::GlobalModel)
    return [check_feasibility(bbl) for bbl in gm.bbls]
end

""" Checks that a BlackBoxLearner.learner has adequate accuracy."""
function check_accuracy(bbc::BlackBoxClassifier)
    return bbc.accuracies[end] >= get_param(bbc, :threshold_accuracy)
end

check_accuracy(bbr::BlackBoxRegressor) = 1. # TODO: use MSE?

function check_sampled(bbl::BlackBoxLearner)
    size(bbl.X, 1) == 0 && throw(OCTHaGOnException(string("BlackBoxLearner ", bbl.name, " must be sampled first.")))
    return 
end

""" 
    get_random_trees(lnr::IAI.Heuristics.RandomForestLearner)
Returns one of the trees of RandomForestLearner. 
"""
function get_random_trees(lnr::IAI.Heuristics.RandomForestLearner)
    trees = []
    for i = 1:lnr.num_trees
        tree = IAI.clone(lnr.inner)
        tree.prb_ = lnr.prb_
        tree.tree_ = lnr.forest_.trees[i]
        push!(trees, tree)
    end
    return trees
end

""" 
    boundify(lnr::OptimalTreeRegressor, X::DataFrame, Y; solver = CPLEX_SILENT)
    boundify(lnr::IAI.Heuristics.RandomForestRegressor, X::DataFrame, Y; solver = CPLEX_SILENT)
    boundify(lnr::OptimalTreeClassifier, X::DataFrame, Y, hypertype::String = "lower"; solver = CPLEX_SILENT)
Returns bounding hyperplanes of an OptimalTreeLearner.
"""
function boundify(lnr::IAI.OptimalTreeRegressor, X::DataFrame, Y; solver = CPLEX_SILENT)
    feas_leaves = find_leaves(lnr)
    leaf_idx = IAI.apply(lnr, X)
    ul_data = Dict()
    for leaf in feas_leaves
        idx = findall(x -> x == leaf, leaf_idx)
        ul_data[-leaf] = u_regress(X[idx,:], Y[idx]; solver = solver) # negative leaf shows upper bounding
        ul_data[leaf] = l_regress(X[idx,:], Y[idx]; solver = solver)
    end
    return ul_data
end

function boundify(lnr::IAI.OptimalTreeClassifier, X::DataFrame, Y, hypertype::String = "lower"; solver = CPLEX_SILENT)
    all_leaves = find_leaves(lnr)
    leaf_idx = IAI.apply(lnr, X)
    data = Dict()
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    for leaf in feas_leaves
        idx = findall(x -> x == leaf, leaf_idx)
        if hypertype == "upper"
            data[-leaf] = u_regress(X[idx,:], Y[idx]; solver = solver)
        elseif hypertype == "lower"
            data[leaf] = l_regress(X[idx,:], Y[idx]; solver = solver)
        end
    end
    return data
end

function boundify(lnr::IAI.Heuristics.RandomForestRegressor,
                  X::DataFrame, Y; solver = CPLEX_SILENT)
    trees = get_random_trees(lnr)
    data = Dict()
    for i=1:length(trees)
        ul_data = Dict()
        tree = trees[i]
        all_leaves = find_leaves(tree)
        leaf_idx = IAI.apply(tree, X)
        for leaf in all_leaves
            idx = findall(x -> x == leaf, leaf_idx)
            ul_data[-leaf] = u_regress(X[idx,:], Y[idx]; solver = solver) # negative leaf shows upper bounding
            ul_data[leaf] = l_regress(X[idx,:], Y[idx]; solver = solver)
        end
        data[i] = ul_data
    end
    return data
end

function boundify(lnr::IAI.Heuristics.RandomForestClassifier,
                    X::DataFrame, Y, hypertype::String = "lower"; solver = CPLEX_SILENT)
    trees = get_random_trees(lnr)
    data = Dict()
    for i=1:length(trees)
        ul_data = Dict()
        tree = trees[i]
        all_leaves = find_leaves(tree)
        leaf_idx = IAI.apply(tree, X)
        feas_leaves = [j for j in all_leaves if Bool(IAI.get_classification_label(tree, j))]
        for leaf in feas_leaves
            idx = findall(x -> x == leaf, leaf_idx)
            if hypertype == "upper"
                ul_data[-leaf] = u_regress(X[idx,:], Y[idx]; solver = solver)
            elseif hypertype == "lower"
                ul_data[leaf] = l_regress(X[idx,:], Y[idx]; solver = solver)
            end
        end 
        data[i] = ul_data
    end
    return data
end


function learn_constraint_IAI!(lnr::Union{IAI.OptimalTreeRegressor, IAI.OptimalTreeClassifier, 
    IAI.Heuristics.RandomForestRegressor}, bbc::BlackBoxClassifier; kwargs...)

    IAI.set_params!(lnr; minbucket = 
    maximum([2*length(bbc.vars)/length(bbc.Y), lnr.minbucket]))
    if bbc.equality # Equalities are approximated more aggressively.
        IAI.set_params!(lnr; max_depth = 6, ls_num_tree_restarts = 30, fast_num_support_restarts = 15)
    end
    IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)

    #(X_train, y_train), (X_test, y_test) = train_test_split(bbc.X, bbc.Y .>= 0; at = 0.67)
    X = DataFrame(bbc.X)
    Y = bbc.Y .>= 0
    #X_test = DataFrame(X_test)
    if check_feasibility(bbc) || get_param(bbc, :ignore_feasibility)
        lnr, score, (A_all_u, t_all_u, A_all_l, t_all_l) = learn_from_data_ro!(X, Y, lnr, bbc; fit_classifier_kwargs(; kwargs...)...)
        #score = IAI.score(lnr, X_test, y_test)
        
        bbc.ro_data = (A_all_u, t_all_u, A_all_l, t_all_l)
        
        return lnr, score, Dict(kwargs)
        # push!(bbc.learners, lnr)
        # push!(bbc.accuracies, IAI.score(lnr, bbc.X, bbc.Y .>= 0)) # TODO: add ability to specify criterion. 
        # push!(bbc.learner_kwargs, Dict(kwargs))
    else
        throw(OCTHaGOnException("Not enough feasible samples for BlackBoxClassifier " * string(bbc.name) * "."))
    end
end


# function learn_constraint_SVM!(bbc::BlackBoxClassifier; kwargs...)
#     if check_feasibility(bbc) || get_param(bbc, :ignore_feasibility)

#         #score = IAI.score(lnr, bbc.X, bbc.Y .>= 0)

#         (X_train, y_train), (X_test, y_test) = train_test_split(bbc.X, bbc.Y .>= 0; at = 0.67)

#         (β0, β) = svm(Matrix(X_train), 1*(y_train .>= 0))
        
#         y_pred = 1*(Matrix(X_test) * β .+ β0 .> 0.5 )

#         score = roc_auc_score(y_test, y_pred)

#         lnr = SVM_Classifier(β0=β0, β=β)

#         return lnr, score, Dict(kwargs)
#     else
#         throw(OCTHaGOnException("Not enough feasible samples for BlackBoxClassifier " * string(bbc.name) * "."))
#     end
# end



"""
    learn_constraint!(bbl::Union{GlobalModel, BlackBoxLearner, Array}; kwargs...)

Constructs a constraint tree from a BlackBoxLearner and dumps in bbo.learners.
The models used for constraint approximation are specified in bbc.alg_list 
or in the argument `algs` (if the argument algs is provided, then this argument
is used, otherwise, will be used).
If more than 1 models are specified, then this function
finds the best model (in terms of AUC) and uses that to approximate the
constraint
Arguments:
    bbl: OCT structs (GM, bbl, Array{bbl})
    algs: A list of the different models that will be used for learning.
          Possible values are "OCT","CART","RF".
          Example use: algs=["CART", "RF", "RF"]
          If this argument is not specified, then
          its value is determined by the field bbc.alg_list
    kwargs: arguments for learners and fits.
"""
function learn_constraint!(bbc::BlackBoxClassifier, algs::Union{Nothing, Array{String}} = nothing; kwargs...)
    check_sampled(bbc)
    set_param(bbc, :reloaded, false) # Makes sure that we know trees are retrained. 

    if isnothing(algs) 
        algs = bbc.alg_list
    end
    
    best_score = -Inf
    best_alg_name = nothing
    best_model = nothing

    for alg in algs
        
        if !haskey(LEARNER_DICT["classification"], alg)
            throw(OCTHaGOnException("$(alg) model is not supported for classification."))
        end

        try
            ts = time()
            if alg in ["CART", "OCT"]
                # Use IAI
                lnr = LEARNER_DICT["classification"][alg]()
                lnr, score, args = learn_constraint_IAI!(lnr, bbc; kwargs...)
            else
                lnr = LEARNER_DICT["classification"][alg]()
                lnr, score = learn_from_data!(bbc.X, bbc.Y, lnr; equality = bbc.equality)
                args = kwargs
            end

            append!(bbc.learner_performance, Dict(
                "model" => alg,
                "type" => "classification", 
                "constr" => bbc.name, 
                "n" => length(bbc.vars),
                "score" => score,
                "time" => time()-ts)
            )

            @info "Trained $(alg) with ACCURACY=$(score)"
            if score >= best_score
                best_alg_name = alg
                best_score  = score
                best_model = (lnr, score, args)
            end
        catch
            @warn "Model $(alg) failed"
        end
    end

    lnr, score, args = best_model
    @info "Best model: $(best_alg_name) with ACCURACY=$(score)"
    
    push!(bbc.learners, lnr)
    push!(bbc.accuracies, score) # TODO: add ability to specify criterion. 
    push!(bbc.learner_kwargs, args)

    return
end

function learn_constraint!(bbr::BlackBoxRegressor, threshold::Pair = Pair("reg", nothing), algs::Union{Nothing, Array{String}} = nothing;  kwargs...)
    classifications = ["upper", "lower", "upperlower"]
    check_sampled(bbr)
    get_param(bbr, :reloaded) && set_param(bbr, :reloaded, false) # Makes sure that we know trees are retrained. 

    if isnothing(algs) 
        algs = bbr.alg_list
    end
    
    best_score = -Inf
    best_alg_name = nothing
    best_model = nothing

    for alg in algs 

        try
            if bbr.convex && !bbr.equality
                return # If convex, don't train a tree!
            elseif threshold.first in classifications
                if alg != "OCT" && alg != "CART"
                    @warn("$(alg) regressor attempts to be used as classifier. Skipping")
                    continue
                end
                lnr = base_classifier()
                IAI.set_params!(lnr; minbucket = 
                    maximum([2*length(bbr.vars)/length(bbr.Y), lnr.minbucket]), 
                    classifier_kwargs(; kwargs...)...)
                ul_data = Dict()
                if threshold.first == "upper" # Upper bounding classifier with upper bounds in leaves
                    lnr, score = learn_from_data!(bbr.X, bbr.Y .<= threshold.second, lnr; fit_classifier_kwargs(; kwargs...)...)
                    merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, "upper"))
                elseif threshold.first == "lower" # Lower bounding classifier with lower bounds in leaves
                    lnr, score = learn_from_data!(bbr.X, bbr.Y .>= threshold.second, lnr; fit_classifier_kwargs(; kwargs...)...)
                    merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, "lower"))
                elseif threshold.first == "upperlower" # Upper bounding classifier with bounds in each leaf
                    lnr, score = learn_from_data!(bbr.X, bbr.Y .<= threshold.second, lnr; fit_classifier_kwargs(; kwargs...)...)
                    merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, "lower"))
                    merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, "upper"))
                end
                push!(bbr.learners, lnr);
                push!(bbr.learner_kwargs, Dict(kwargs))
                push!(bbr.thresholds, threshold)
                push!(bbr.ul_data, ul_data)
                push!(bbr.accuracies, IAI.score(lnr, bbr.X, bbr.Y .>= 0))
                return
            elseif alg ∉ ["OCT", "CART"]
                ts = time()

                lnr = LEARNER_DICT["regression"][alg](dependent_var = bbr.dependent_var)

                
                lnr, score = learn_from_data!(bbr.X, bbr.Y, lnr; equality = bbr.equality)

                append!(bbr.learner_performance, Dict(
                    "model" => alg,
                    "type" => "regression", 
                    "constr" => bbr.name, 
                    "n" => length(bbr.vars),
                    "score" => score,
                    "time" => time()-ts)
                )

                @info "Trained $(alg) with R2=$(score)"
                if score >= best_score
                    best_alg_name = alg
                    best_score  = score
                    best_model = (lnr, score, Dict(kwargs))
                end
                
                push!(bbr.learners, lnr);
                push!(bbr.learner_kwargs, Dict(kwargs))
                #push!(bbr.thresholds, threshold)
                #push!(bbr.ul_data, boundify(lnr, bbr.X, bbr.Y))
                push!(bbr.accuracies, score)
            elseif threshold.first == "reg"

                ts = time()

                lnr = LEARNER_DICT["regression"][alg](dependent_var = bbr.dependent_var)
                IAI.set_params!(lnr; minbucket = 
                    maximum([2*length(bbr.vars)/length(bbr.Y), lnr.minbucket]),
                    regressor_kwargs(; kwargs...)...)
                if bbr.equality # Equalities cannot leverage convexity unfortunately...
                    if threshold.second == nothing
                        lnr, score = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)   
                    else
                        idxs = findall(y -> y <= threshold.second, bbr.Y) 
                        lnr, score = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs], lnr; fit_regressor_kwargs(; kwargs...)...)
                    end        
                elseif threshold.second == nothing && bbr.local_convexity < 0.75
                    lnr, score = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)   
                elseif threshold.second == nothing && bbr.local_convexity >= 0.75
                    lnr, score = learn_from_data!(bbr.X, bbr.curvatures .> 0, lnr; fit_regressor_kwargs(; kwargs...)...)             
                elseif bbr.local_convexity < 0.75
                    idxs = findall(y -> y <= threshold.second, bbr.Y) 
                    lnr, score = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs], lnr; fit_regressor_kwargs(; kwargs...)...)
                elseif bbr.local_convexity >= 0.75
                    idxs = findall(y -> y <= threshold.second, bbr.Y) 
                    lnr, score = learn_from_data!(bbr.X[idxs, :], bbr.curvatures[idxs] .> 0, lnr; fit_regressor_kwargs(; kwargs...)...)
                end
                
                append!(bbr.learner_performance, Dict(
                    "model" => alg,
                    "type" => "regression", 
                    "constr" => bbr.name, 
                    "n" => length(bbr.vars),
                    "score" => score,
                    "time" => time()-ts)
                )

                @info "Trained $(alg) with R2=$(score)"
                if score >= best_score
                    best_alg_name = alg
                    best_score  = score
                    best_model = (lnr, score, Dict(kwargs))
                end
                push!(bbr.learners, lnr);
                push!(bbr.learner_kwargs, Dict(kwargs))
                push!(bbr.thresholds, threshold)
                push!(bbr.ul_data, boundify(lnr, bbr.X, bbr.Y))
                push!(bbr.accuracies, score)
            elseif threshold.first == "rfreg"
                lnr = base_rf_regressor()
                bbr.local_convexity < 0.75 || throw(OCTHaGOnException("Cannot use RandomForestRegressor " *
                "on BBR $(bbr.name) since it is almost convex."))
                IAI.set_params!(lnr; minbucket = 
                    maximum([2*length(bbr.vars)/length(bbr.Y), lnr.minbucket]),
                    regressor_kwargs(; kwargs...)...)
                if threshold.second == nothing
                    lnr, score = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)   
                else
                    idxs = findall(y -> y <= threshold.second, bbr.Y) 
                    lnr, score = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs], lnr; fit_regressor_kwargs(; kwargs...)...)
                end
                push!(bbr.learners, lnr);
                push!(bbr.learner_kwargs, Dict(kwargs))
                push!(bbr.thresholds, threshold)
                push!(bbr.ul_data, boundify(lnr, bbr.X, bbr.Y))
                push!(bbr.accuracies, IAI.score(lnr, bbr.X, bbr.Y))
            else
                throw(OCTHaGOnException("$(threshold.first) is not a valid learner type for" *
                    " thresholded learning of BBR $(bbr.name)."))
            end  
            
        catch
            @warn "Model $(alg) failed"
        end
    end

    if threshold.first != "rfreg"
        bbr.learners = [best_model[1]]
        bbr.learner_kwargs = [Dict(kwargs)]
        bbr.accuracies = [best_score]
    end
    return
end

function learn_constraint!(bbl::Array; kwargs...)
   for fn in bbl
        learn_constraint!(fn; kwargs...)
   end
end

learn_constraint!(gm::GlobalModel; kwargs...) = 
    learn_constraint!(gm.bbls; kwargs...)