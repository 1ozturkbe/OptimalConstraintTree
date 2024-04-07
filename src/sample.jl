"""
    truncate_sigfigs(data, digits = 8)

Cuts off more than digits significant figures. 
Used to reduce tree training degeneracies. 
"""
function truncate_sigfigs(data, digits = 7)
    return round.(data, sigdigits = digits)
end

"""
    lh_sample(vars::Array{JuMP.VariableRef, 1}; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
    lh_sample(bbl::BlackBoxLearner; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)

Uniformly Latin Hypercube samples the variables of GlobalModel, as long as all
lbs and ubs are defined.
"""
function lh_sample(vars::Array{JuMP.VariableRef, 1}; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
    bounds = get_bounds(vars)
    check_bounds(bounds)
    n_dims = length(vars)
    if lh_iterations > 0
        plan, _ = LHCoptim(n_samples, n_dims, lh_iterations);
    else
        plan = rand(n_samples, n_dims)
    end
   X = scaleLHC(plan,[(minimum(bounds[var]), maximum(bounds[var])) for var in vars]);
   return DataFrame(truncate_sigfigs(X), string.(vars))
end

function lh_sample(bbl::BlackBoxLearner; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
   return lh_sample(bbl.vars; lh_iterations = lh_iterations, n_samples = n_samples)
end

function choose(large::Int64, small::Int64)
    return Int64(factorial(big(large)) / (factorial(big(large-small))*factorial(big(small))))
end

"""
    boundary_sample(bbl::BlackBoxLearner; fraction::Float64 = 0.5)
    boundary_sample(vars::Array{JuMP.VariableRef, 1}; n_samples = 100, fraction::Float64 = 0.5,
                         warn_string::String = "")

*Smartly* samples the constraint along the variable boundaries.
    NOTE: Because we are sampling symmetrically for lower and upper bounds,
    the choose coefficient has to be less than ceil(half of number of dims).
"""
function boundary_sample(vars::Array{JuMP.VariableRef, 1}; n_samples::Int64 = 100, fraction::Float64 = 0.5,
                         warn_string::String = "")
    bounds = get_bounds(vars);
    check_bounds(bounds);
    n_vars = length(vars);
    vks = string.(vars);
    lbs = DataFrame(Dict(string(key) => minimum(val) for (key, val) in bounds))
    ubs = DataFrame(Dict(string(key) => maximum(val) for (key, val) in bounds))
    n_comb = 0
    if n_vars <= 12
        n_comb = sum(choose(n_vars, i) for i=0:n_vars);
    else
        n_comb = 1e6
    end
    nX = DataFrame(vks .=> [Float64[] for i in vks])
    sample_indices = [];
    if n_comb >= fraction*n_samples 
        @info("Can't exhaustively sample the boundary of Constraint " * string(warn_string) * ".")
        n_comb = 2*n_vars+2; # Everything is double because we choose min's and max's
        choosing = 1;
        while n_comb <= fraction*n_samples
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
    return nX # Note: boundary samples are not truncated for significant figures
end

function boundary_sample(bbl::BlackBoxLearner; fraction::Float64 = 0.5)
    return boundary_sample(bbl.vars, n_samples = get_param(bbl, :n_samples), fraction = fraction,
                           warn_string = bbl.name)
end


"""
    knn_sample(bbl::BlackBoxClassifier; k::Int64 = 10, sample_density = 1e-5, sample_idxs = nothing)

Does KNN and secant method based sampling once there is at least one feasible
    sample to a BlackBoxLearner.
"""
function knn_sample(bbl::BlackBoxClassifier; k::Int64 = 10, sample_density = 1e-5, sample_idxs = nothing, sign = 1)
    if bbl.feas_ratio == 0. || bbl.feas_ratio == 1.0
        throw(OCTHaGOnException("Constraint " * string(bbl.name) * " must have at least one feasible or
                            infeasible sample to be KNN-sampled!"))
    end
    vks = string.(bbl.vars)
    df = DataFrame(vks .=> [Float64[] for i in vks])
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k=k);
    feas_class = classify_patches(bbl, idxs);
    negatives = findall(x -> x .< 0, bbl.Y)
    if !isnothing(sample_idxs)
        negatives = intersect(negatives, sample_idxs)
    end
    for i = 1:length(negatives) # This loop is for making sure that every possible root is sampled only once.
        if feas_class[negatives[i]] == "mixed"
            nodes = [idxs[negatives[i]][j] for j=1:length(idxs[negatives[i]]) 
                            if (bbl.Y[idxs[negatives[i]][j]] >= 0 && dists[negatives[i]][j] >= sample_density)]
            push!(nodes, negatives[i])
            np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
            append!(df, np)
        end
    end
    return truncate_sigfigs(df)
end

function describe_leaves(leaf_data, root, current_bounds)
    
    if DecisionTree.is_leaf(root)
        #print(current_bounds)
        push!(leaf_data, Dict("bounds"  => reduce(hcat, deepcopy(current_bounds))', "val" => mean(root.values), "n_vals"=>length(root.values)))
    else 
        tmp_bounds = copy(current_bounds)
        tmp_bounds[root.featid][2] = root.featval
        describe_leaves(leaf_data, root.left, tmp_bounds)
        
        tmp_bounds = copy(current_bounds)
        tmp_bounds[root.featid][1] = root.featval
        describe_leaves(leaf_data, root.right, tmp_bounds)
    end
end


function refined_derivative_sampling(bbl::BlackBoxLearner, total_samples=100, depth=4)

    try
        leaf_data = []
        #og_bounds = [[-1e6, 1e6] for _ =1:length(bbl.vars)]

        bounds = get_bounds(bbl.vars)
        og_bounds = [bounds[k] for k in bbl.vars]
        
        grads = [bbl.g(Vector(bbl.X[i,string.(bbl.vars)])) for i=1:size(bbl.X,1)]

        valid_grad_ids = filter((i)->!isnan(sum(grads[i])) && !isinf(sum(grads[i])),1:length(grads))
        grads = grads[valid_grad_ids]
        grad_norms = [norm(grads[i,:]) for i=1:size(grads,1)]

        dr = DecisionTree.DecisionTreeRegressor(max_depth=depth,min_samples_leaf=2)
        DecisionTree.fit!(dr, Matrix(bbl.X[valid_grad_ids,:]), grad_norms)

        describe_leaves(leaf_data, dr.root, og_bounds)

        leaf_data = filter((x)->norm(x["bounds"][:,2]-x["bounds"][:,1])>1e-1, leaf_data)

        total_score = 0
        for i=1:length(leaf_data)
            leaf_data[i]["score"] = leaf_data[i]["val"]/leaf_data[i]["n_vals"]
            total_score += leaf_data[i]["score"]
        end

        Xs = []
        for i=1:length(leaf_data)
            n_samples = floor(Int32, total_samples *(leaf_data[i]["score"])/total_score)
            if n_samples > 0
                xr = Random.rand(n_samples,2)
                xr = xr ./ sum(xr,dims=2)
                sample = xr*leaf_data[i]["bounds"]'
                if length(Xs) == 0
                    Xs = sample
                else
                    Xs = vcat(Xs, sample)
                end
            end
        end

        Xs = DataFrame(Xs, string.(bbl.vars))

        return Xs 
    catch
        println("Couldn't do refined sampling")
        return DataFrame() 
    end

end



"""
    uniform_sample_and_eval!(bbl::Union{BlackBoxLearner, GlobalModel, Array{BlackBoxLearner}};
                              boundary_fraction::Float64 = 0.5,
                              lh_iterations::Int64 = 0)

Uniform samples and evaluates a BlackBoxLearner.
Furthermore, sets the big-M value. 
Keyword arguments:
    boundary_fraction: maximum ratio of boundary samples
    lh_iterations: number of GA populations for LHC sampling (0 is a random LH.)
"""
function uniform_sample_and_eval!(bbl::BlackBoxLearner;
                          boundary_fraction::Float64 = 0.5,
                          lh_iterations::Int64 = 0, sample_density = 1.e-5, opt_sampling::Bool=false, gm::Union{GlobalModel,Nothing}=nothing)
    @assert size(bbl.X, 1) == 0 # TODO: fix this w.r.t. data-driven constraints. 
    vks = string.(bbl.vars)
    n_dims = length(vks);
    check_bounds(get_bounds(bbl))
    df = boundary_sample(bbl, fraction = boundary_fraction)
    eval!(bbl, df)
    df = lh_sample(bbl, lh_iterations = lh_iterations, n_samples = get_param(bbl, :n_samples) - size(df, 1))
    eval!(bbl, df);

    og_num_samples = get_param(bbl, :n_samples)
    
    for (num_samples, depth) in [(og_num_samples,4),(og_num_samples,5)]
        df = refined_derivative_sampling(bbl, og_num_samples,depth )
        if (size(df,1) >0)
            eval!(bbl, df)
        end
    end

    bbl.max_Y = maximum(filter(!isinf, bbl.Y))
    bbl.min_Y = minimum(filter(!isinf, bbl.Y))
    
    if opt_sampling
        @info("Starting opt sampling. Samples before $(length(bbl.Y))")
        opt_sample(gm, bbl)
        @info("Samples after $(length(bbl.Y))")
    end

    if bbl isa BlackBoxClassifier
        if bbl.feas_ratio == 1.0
            @info(string(bbl.name) * " was not KNN sampled since it has no infeasible samples.")
        elseif bbl.feas_ratio == 0.0
            throw(OCTHaGOnException(string(bbl.name) * " has zero feasible samples. " *
                               "Please find at least one feasible sample, seed the data and KNN sample."))
        else
            df = knn_sample(bbl, k= maximum([10, 2*length(bbl.vars) + 1]), sample_density = sample_density)
            if size(df, 1) > 0
                eval!(bbl, df)
            end
        end
    end
    bbl.max_Y = maximum(filter(!isinf, bbl.Y))
    bbl.min_Y = minimum(filter(!isinf, bbl.Y))
    # TODO: update these mx/min Y more frequently. 
    # Setting vague bounds for the dependent variable. 
    # This is the only big-M required in the formulation, adding 50% margin to all observed Y values. 
    if bbl isa BlackBoxRegressor && !isnothing(bbl.dependent_var)
        max_Y = bbl.max_Y
        min_Y = bbl.min_Y
        lower_margined_bound = min_Y - (max_Y - min_Y)/2
        upper_margined_bound = max_Y + (max_Y - min_Y)/2
        if !JuMP.has_lower_bound(bbl.dependent_var)
            JuMP.set_lower_bound(bbl.dependent_var, lower_margined_bound)
        end
        if !JuMP.has_upper_bound(bbl.dependent_var)
            JuMP.set_upper_bound(bbl.dependent_var, upper_margined_bound)
        end
        for ll in bbl.lls
            if !JuMP.has_lower_bound(ll.dependent_var)
                JuMP.set_lower_bound(ll.dependent_var, lower_margined_bound)
            end
            if !JuMP.has_upper_bound(ll.dependent_var)
                JuMP.set_upper_bound(ll.dependent_var, upper_margined_bound)
            end
        end
    end

    if get_param(gm, :oct_sampling) && (bbl isa BlackBoxClassifier) 
        # Saving code 
        pre_len = 1:length(bbl.Y)
        #####################################################################
        df = oct_sampling(bbl)
        if (size(df,1) >0)
            eval!(bbl, df)
        end

        df_c = copy(bbl.X)
        df_c[!, "Y"] = bbl.Y
        df_c[!, "oct"] = (1:size(df_c, 1)) .>= pre_len
    end

    return 
end

function uniform_sample_and_eval!(bbls::Array{BlackBoxLearner}; lh_iterations = 0, sample_density = 1e-5) 
    for bbl in bbls 
        uniform_sample_and_eval!(bbl, lh_iterations = lh_iterations, sample_density = sample_density)
    end
    return
end

function uniform_sample_and_eval!(gm::GlobalModel)
    @info "Sampling..."
    uniform_sample_and_eval!(gm.bbls; lh_iterations = get_param(gm, :lh_iterations), sample_density = get_param(gm, :sample_density))
    return
end

"""
    last_leaf_sample(bbl::BlackBoxLearner)

Gets Latin Hypercube samples that fall in the leaf of the last solution.
"""
function last_leaf_sample(bbc::BlackBoxClassifier, n_samples = get_param(bbc, :n_samples))
    isempty(bbc.active_leaves) &&         
        throw(OCTHaGOnException("BBC $(bbc.name) needs to be optimized first, to figure out its active leaves."))
    if !bbc.equality
        last_leaf = bbc.active_leaves[1]
        idxs = findall(x -> x .>= 0.5, IAI.apply(bbc.learners[end], bbc.X) .== last_leaf)
        lbs = [minimum(col) for col in eachcol(bbc.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbc.X[idxs, :])]
        plan, _ = LHCoptim(n_samples, length(bbc.vars), 3);
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        return DataFrame(truncate_sigfigs(X), string.(bbc.vars))
    else
        for last_leaf in bbc.active_leaves
            idxs = findall(x -> x .>= 0.5, IAI.apply(bbc.learners[end], bbc.X) .== last_leaf)
            lbs = [minimum(col) for col in eachcol(bbc.X[idxs, :])]
            ubs = [maximum(col) for col in eachcol(bbc.X[idxs, :])]
            plan, _ = LHCoptim(Int(ceil(n_samples/2)), length(bbc.vars), 3);
            X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
            return DataFrame(truncate_sigfigs(X), string.(bbc.vars))
        end
    end
end

function last_leaf_sample(bbr::BlackBoxRegressor, n_samples = get_param(bbr, :n_samples))
    if isempty(bbr.active_leaves)
        throw(OCTHaGOnException("BBR $(bbr.name) needs to be optimized first, to figure out its active leaves."))
    elseif length(bbr.active_trees) == 2
        upper_leaf, lower_leaf = sort(bbr.active_leaves)
        upper_leafneighbor = [];
        lower_leafneighbor = [];
        for (tree_idx, threshold) in bbr.active_trees
            if threshold.first == "upper"
                append!(upper_leafneighbor, IAI.apply(bbr.learners[tree_idx], bbr.X) .== -upper_leaf)
            elseif threshold.first == "lower"
                append!(lower_leafneighbor, IAI.apply(bbr.learners[tree_idx], bbr.X) .== lower_leaf)
            end
        end
        idxs =  findall(x -> x .>= 0.5, upper_leafneighbor .* lower_leafneighbor)
        if length(idxs) == 0
            @info("No points in $(bbr.name) in the intersection of trees. Widening sampling. ")
            idxs =  findall(x -> x .>= 0.5, upper_leafneighbor + lower_leafneighbor)
        end
        lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
        plan, _ = LHCoptim(n_samples, length(bbr.vars), 3);
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        return DataFrame(truncate_sigfigs(X), string.(bbr.vars))
    elseif length(bbr.active_trees) == 1
        leaf = bbr.active_leaves[1]
        tree = bbr.learners[collect(keys(bbr.active_trees))[1]]
        idxs = findall(x -> x .== leaf, IAI.apply(tree, bbr.X))
        lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
        plan, _ = LHCoptim(n_samples, length(bbr.vars), 3);
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        return DataFrame(truncate_sigfigs(X), string.(bbr.vars))
    else 
        throw(OCTHaGOnException("No active trees found in BBR $(bbr.name) while attempting to leaf-sample."))
    end
end

"""
    feasibility_sample(gm::GlobalModel)
    feasibility_sample(bbc::BlackBoxClassifier, n_samples::Int64 = get_param(bbc, :n_samples))

Equally LH samples the leaves of BBC with not enough feasible samples. 
"""
function feasibility_sample(bbc::BlackBoxClassifier, n_samples::Int64 = get_param(bbc, :n_samples))
    minbucket = minimum([Int64(round(bbc.feas_ratio .* size(bbc.X, 1))), Int64(floor(0.05*size(bbc.X, 1)))])
    lnr = base_classifier()
    IAI.set_params!(lnr, minbucket = minbucket)
    lnr, score = learn_from_data!(bbc.X, bbc.Y .>= 0, lnr; fit_classifier_kwargs()...)
    all_leaves = find_leaves(lnr)
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    orig_feasratio = copy(bbc.feas_ratio)
    leaf_idxs = IAI.apply(lnr, bbc.X)
    for leaf in feas_leaves
        idxs = findall(x -> x .>= 0.5, leaf_idxs .== leaf)
        lbs = [minimum(col) for col in eachcol(bbc.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbc.X[idxs, :])]
        plan, _ = LHCoptim(Int64(ceil(n_samples / length(feas_leaves))), length(bbc.vars), 3);  
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        eval!(bbc, DataFrame(X, string.(bbc.vars)))
    end    
    if bbc.feas_ratio >= get_param(bbc, :threshold_feasibility)
        @info("BBC $(bbc.name) has passed threshold feasibility $(round(get_param(bbc, :threshold_feasibility), sigdigits=3)), " * 
              "from $(round(orig_feasratio, sigdigits=3)) to $(round(bbc.feas_ratio, sigdigits=3)).")
    elseif bbc.feas_ratio > orig_feasratio
        @info("BBC $(bbc.name) has improved feasibility from $(round(orig_feasratio, sigdigits=3)) to $(round(bbc.feas_ratio, sigdigits=3)), " * 
                "but still not over threshold $(round(get_param(bbc, :threshold_feasibility), sigdigits=3)).")
    else
        @info("BBC $(bbc.name) feasibility improvement failed! Please check your functions.")
    end
    return
end

function feasibility_sample(gm::GlobalModel)
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier && bbl.feas_ratio <= get_param(bbl, :threshold_feasibility)
            feasibility_sample(bbl)
        end
    end
    return
end

""" 
    upper_bound_sample(bbr::BlackBoxRegressor)

Samples within the feasible regions of the active upper bounding tree.
"""
function upper_bound_sample(bbr::BlackBoxRegressor)
    @assert !isempty(bbr.active_trees) # We must have optimized at least once over the old tree. 
    active_upper_idx = active_upper_tree(bbr)
    lnr = bbr.learners[active_upper_idx]
    @assert lnr isa IAI.OptimalTreeClassifier
    all_leaves = find_leaves(lnr)
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    leaf_idxs = IAI.apply(lnr, bbr.X)
    n_samples = get_param(bbr, :n_samples)
    for leaf in feas_leaves
        idxs = findall(x -> x == leaf, leaf_idxs)
        lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
        plan, _ = LHCoptim(Int64(ceil(n_samples / length(feas_leaves))), length(bbr.vars), 3);  
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        eval!(bbr, DataFrame(X, string.(bbr.vars)))
    end
    return
end


function opt_sample(gm::GlobalModel, bbl::BlackBoxLearner)

    # Cluster already sampled points 
    Xc = kmeans(Matrix(bbl.X)', length(bbl.vars)).centers';
    id_map = Dict(string(c) => i for (i,c) in enumerate(bbl.vars));
    #bbl_id = filter(x-> x[2] == bbl, [(i, bbl) for (i, bbl) in enumerate(gm.bbls)])[1][1]
    
    
    # for bb in gm.bbls 
    #     push!(Xo, deepcopy(bbl.X))
    #     push!(Yo, deepcopy(bbl.Y))
    #     push!(Fg, deepcopy(bbl.feas_gap))
    #     push!(G, deepcopy(bbl.soldict))
    # end

    # For every cluster, call the optimization sampling method
    for i in 1:size(Xc, 1)
        #bbl_old = deepcopy(bbl)
        #Xo, Yo, Fg, G = deepcopy(bbl.X), deepcopy(bbl.Y), deepcopy(bbl.feas_gap), deepcopy(bbl.gradients)
        Xo, Yo, Fg, G = [], [], [], []
        for bb in gm.bbls 
            push!(Xo, deepcopy(bb.X))
            push!(Yo, deepcopy(bb.Y))
            push!(Fg, deepcopy(bb.feas_gap))
            push!(G, deepcopy(bb.gradients))
        end

        Sh, Cs, Sd = deepcopy(gm.solution_history), deepcopy(gm.cost), deepcopy(gm.soldict)

        #println("Clustering iteration $(i)")
        x0 = DataFrame(Dict(string(p[1]) => rand(Uniform(p[2][1], p[2][2])) for p in OCTHaGOn.get_bounds(gm)))
        for col in names(x0)
            if haskey(id_map, col)
                x0[1, col] = Xc[i, id_map[col]]
            end
        end
        try
            old_points = length(bbl.Y)
            X_s = opt_sample_helper(gm, bbl, x0)

            for (i, bb) in enumerate(gm.bbls)
                bb.X, bb.Y, bb.feas_gap, bb.gradients = Xo[i], Yo[i], Fg[i], G[i]
            end
            gm.solution_history, gm.cost, gm.soldict = Sh, Cs, Sd
            eval!(bbl, X_s)
            new_points = length(bbl.Y)
            #@info("Cluster $(i): $(new_points-old_points) new points")
        catch 
            #@info("Skipping cluster $(i)")
            for (i, bb) in enumerate(gm.bbls)
                bb.X, bb.Y, bb.feas_gap, bb.gradients = Xo[i], Yo[i], Fg[i], G[i]
            end
            gm.solution_history, gm.cost, gm.soldict = Sh, Cs, Sd
        end
    end
end 

"""
Samples close to the constraint boundary using optimization.
The sampling is done only on feasible points
"""
function opt_sample_helper(gmo::GlobalModel, bblo::BlackBoxLearner, xstart::DataFrame; max_iterations=50, kwargs...)
    
    # bbl_id = filter(x-> x[2] == bblo, [(i, bbl) for (i, bbl) in enumerate(gmo.bbls)])[1][1]
    bbl_id = nothing 
    for (i, bbl) in enumerate(gmo.bbls)
        if bbl == bblo 
            bbl_id = i
        end
    end

    #gm = deepcopy(gmo)
    gm = gmo
    # Initialization
    clear_tree_constraints!(gm) 
    bbls = gm.bbls
    vars = gm.vars
    bounds = get_bounds(vars)

    #gm.solution_history = DataFrames()

    new_samples = DataFrame()

    # Update descent algorithm parameters
    if !isempty(kwargs) 
        for item in kwargs
            set_param(gm, item.first, item.second)
        end
    end

    # Checking for a nonlinear objective
    obj_bbl = gm.bbls[bbl_id]
#     if gm.objective isa VariableRef
#         obj_bbl = filter(x -> x.dependent_var == gm.objective, 
#                             [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor])
#         if isempty(obj_bbl)
#             obj_bbl = nothing
#         elseif length(obj_bbl) == 1
#             obj_bbl = obj_bbl[1]
#             vars = [var for var in vars if var != obj_bbl.dependent_var]
#             bbls = [bbl for bbl in gm.bbls if bbl != obj_bbl]
#         elseif length(obj_bbl) > 1
#             throw(OCTHaGOnException("GlobaModel $(gm.name) has more than one BlackBoxRegressor" *
#                                " on the objective variable."))
#         end
#     end 

    grad_shell = DataFrame(string.(vars) .=> [Float64[] for i=1:length(vars)])

    # Add relaxation variables
    add_relaxation_variables!(gm)

    # Final checks
    obj_gradient = copy(grad_shell)
#     if isnothing(obj_bbl)
#         if gm.objective isa VariableRef
#             append!(obj_gradient, DataFrame(string.(gm.objective) => 1), cols = :subset)
#         elseif gm.objective isa JuMP.GenericAffExpr
#             append!(obj_gradient, DataFrame(Dict(string(key) => value for (key,value) in gm.objective.terms)), cols = :subset)
#         else
#             @warn "Type of objective $(gm.objective) is unsupported."
#         end
#         obj_gradient = coalesce.(obj_gradient, 0)
#     end
    if isempty(obj_gradient)
        obj_gradient = DataFrame(string.(vars) .=> zeros(length(vars)))
    end

    # x0 initialization, and actual objective computation
    #x0 = DataFrame(gm.solution_history[end, :])
    x0 = xstart
    sol_vals = x0[:,string.(vars)]
    if !isnothing(obj_bbl)
        actual_cost = obj_bbl(x0)
        push!(gm.cost, actual_cost[1])
        x0[:, string(gm.objective)] = actual_cost
        sol_vals = x0[:,string.(vars)]
        feas_gap(gm, x0) # Checking feasibility gaps with updated objective
        append!(gm.solution_history, x0) # Pushing last solution
    end

    # Descent direction, counting and book-keeping
    d = @variable(gm.model, [1:length(vars)])
    var_diff = []
    max_diff = 0
    for key in vars
        push!(var_diff, maximum(bounds[key]) - minimum(bounds[key]))
        if !isinf(var_diff[end])
            max_diff = maximum([var_diff[end], max_diff])
        end
    end
    if any(isinf.(var_diff))
        @warn "Unbounded variables detected. " *
              "PGD may fail to converge to a local minimum."
        replace!(var_diff, Inf => max_diff)
    end
    replace!(var_diff, 0 => 1) # Sanitizing fixed values. 

    ct = 0
    d_improv = 1e5
    prev_feas = false
    feas = is_feasible(gm)
    abstol = get_param(gm, :abstol)
    tighttol = get_param(gm, :tighttol)
    #max_iterations = get_param(gm, :max_iterations)
    step_penalty = get_param(gm, :step_penalty)
    equality_penalty = get_param(gm, :equality_penalty)
    step_size = get_param(gm, :step_size) 
    decay_rate = get_param(gm, :decay_rate)

    # WHILE LOOP
    # @info("Starting projected gradient descent...")
    while (!feas || !prev_feas ||
        (prev_feas && feas && d_improv >= get_param(gm, :abstol)) ||
        ct == 0) && ct < max_iterations
        prev_obj = gm.cost[end]
        constrs = []
        ct += 1
        if step_penalty == 0
            step_penalty = maximum([1e3, 1e4*abs(prev_obj)])
        end
        if equality_penalty == 0
            equality_penalty = 100*step_penalty
        end
    
        # Linear objective gradient and constraints
        if !isnothing(obj_bbl)
            update_gradients(obj_bbl, [length(obj_bbl.Y)])
            obj_gradient = copy(grad_shell)
            append!(obj_gradient, 
                DataFrame(obj_bbl.gradients[end,:]), cols = :subset) 
            obj_gradient = coalesce.(obj_gradient, 0)
            # Update objective constraints
            # push!(constrs, @constraint(gm.model, sum(Array(obj_gradient[end,:]) .* d) + 
            #                   obj_bbl.dependent_var >= obj_bbl.Y[end]))
        end

        # Constraint evaluation
        errors = 0
        for bbl in bbls
            update_gradients(bbl, [length(bbl.Y)])
            constr_gradient = copy(grad_shell)
            append!(constr_gradient, DataFrame(bbl.gradients[end,:]), cols = :subset)
            constr_gradient = coalesce.(constr_gradient, 0)
            if bbl isa BlackBoxClassifier
                error_diff = maximum([tighttol, abs(bbl.Y[end])])
                if bbl.equality
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] <= error_diff * bbl.relax_var))
                    push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                    errors += bbl.relax_var^2
                elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                    push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                    errors += bbl.relax_var^2
                else # feasible to zero tolerance
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] >= 0))
                end
            elseif bbl isa BlackBoxRegressor

                if isnothing(bbl.dependent_var)
                    error_diff = maximum([tighttol, abs(bbl.optima[end] - bbl.actuals[end])])
                    if bbl.equality
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + error_diff * bbl.relax_var >= bbl.Y[end]))
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) <= bbl.Y[end] + error_diff * bbl.relax_var))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    else # feasible to zero tolerance
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) >= bbl.Y[end]))
                    end
                else
                    error_diff = maximum([tighttol, abs(bbl.optima[end] - bbl.actuals[end])])
                    if bbl.equality
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var + error_diff * bbl.relax_var >= bbl.Y[end]))
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var <= bbl.Y[end] + error_diff * bbl.relax_var))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    else # feasible to zero tolerance
                        push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var >= bbl.Y[end]))
                    end
                end
            end
        end

        # Initializing descent direction
        push!(constrs, @constraint(gm.model, d .== vars .- Array(sol_vals[end,:])))
        if feas # Small gradient step
            push!(constrs, @constraint(gm.model, sum((d ./ var_diff).^2) <= 
                step_size/exp(decay_rate*(ct-1)/max_iterations)))
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d) + equality_penalty*errors)
        else    # Projection step
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d) + step_penalty*sum((d ./ var_diff).^2) + equality_penalty*errors)
        end

        # Optimizing the step, and finding next x0
        optimize!(gm.model)

        try
            x0 = DataFrame(string.(vars) .=> getvalue.(vars))
            if !isnothing(obj_bbl)
                tmp = obj_bbl(x0)
                x0[!, string(gm.objective)] = tmp
                # println(tmp)
            end
            sol_vals = x0[:,string.(vars)]
            feas_gap(gm, x0)
            append!(gm.solution_history, x0)

            # Measuring improvements
            if !isnothing(obj_bbl)
                push!(gm.cost, gm.solution_history[end, string(gm.objective)])
            else
                push!(gm.cost,  JuMP.getvalue(gm.objective))
            end
            d_improv = gm.cost[end-1] - gm.cost[end]

            # Feasibility check
            prev_feas = feas
            feas = is_feasible(gm)

            if feas
                append!(new_samples, x0)
            end

            # Saving solution dict for JuMP-style recovery.
            # TODO: do not regenerate soldict unless final solution. 
            gm.soldict = Dict(key => JuMP.getvalue.(gm.model[key]) for (key, value) in gm.model.obj_dict)
            for con in constrs
                delete(gm.model, con)
            end
        catch
            for con in constrs
                delete(gm.model, con)
            end
            break
        end
    end

    fincost = round(gm.cost[end], digits = -Int(round(log10(abstol))))
    # Termination criteria
    # if ct >= max_iterations
    #     @info "Max iterations ($(ct)) reached."
    #     if prev_feas && feas
    #         @info "Solution is not converged to tolerance $(abstol)!" 
    #     elseif (feas && !prev_feas) && (abs(d_improv) <= 100*abstol)
    #         @info "Solution is feasible and likely cycling, but the solution is close. Reduce step size and descend again. "
    #     elseif (!feas && prev_feas) && (abs(d_improv) <= 100*abstol)
    #         @info "Solution is infeasible and likely cycling, but the solution is close. Reduce step size and descend again. "
    #     elseif ((feas && !prev_feas) || (prev_feas && !feas))
    #         @info "Solution is likely cycling, with > $(abstol) changes in cost. Reduce step size and descend again."
    #     elseif !feas && !prev_feas
    #         @info "Solution is infeasible to tolerance $(tighttol)."
    #     end
    #     @info("Final cost is $(fincost).")
    # else 
    #     @info("PGD converged in $(ct) iterations!")
    #     @info("The optimal cost is $(fincost).")
    # end

    # Reverting objective, and deleting vars
    @objective(gm.model, Min, gm.objective)
    delete(gm.model, d)
    return new_samples
end

function har_sample_helper(A, b, x_start)
    
    max_tries = 10
    
    success = false
    lambdas = nothing
    dir = nothing

    for i=1:max_tries

        # Generate a random direction
        dir = rand(length(x_start))
        dir = dir ./ norm(dir)

        # Calculate maximum/minimum lambdas from
        # all hyperplanes (i.e. constants such that x_start+lambda*dir
        # lies on the hyperplane)
        lambdas = (b.-A*x_start) ./ (A*dir)

        if !any(isinf.(lambdas)) && length(lambdas[lambdas.<0]) >0 && length(lambdas[lambdas.>0]) >0
            success = true
            break
        end
    end

    if !success
        error("Problem with HAR sampling")
    end

    # Minimum and maximum values for lambda
    lam_minus = 1.05*maximum(lambdas[lambdas.<0])
    lam_plus = 0.95*minimum(lambdas[lambdas.>0])

    # Generate a random lambda between the values
    lambda = rand(Uniform(lam_minus, lam_plus))

    # Final sample point
    p = x_start + lambda*dir
    
    return p
end

function har_sample(A, b, x_start, N_samples = 10)
    
    X = []
    
    i = 1
    max_tries = 50
    tries = 0
    
    while i<= N_samples && tries <= max_tries
        
        # HAR sample
        p = har_sample_helper(A, b, x_start)
        
        if  !all(A*p .<= b)
            continue
            tries += 1
        end
        
        tries = 0
        push!(X, p)
        x_start = p
        i += 1
    end
    
    return X
end

function oct_sampling(bbl::BlackBoxClassifier, sampling_factor=0.5)
    
    # try
        all_idx = collect(1:size(bbl.X, 1))
        upper_dicts = []
        lower_dicts = []
        lnrs = []

        for i= 1:8
        #     try
                println("OCT sampling $(i)")
                lnr = OCTHaGOn.LEARNER_DICT["classification"]["OCT"]()
                sub_idx = StatsBase.sample(all_idx, trunc(Int, size(bbl.X,1)*0.8))
                lnr, score = OCTHaGOn.learn_from_data!(copy(bbl.X[sub_idx,:]), 1.0*(copy(bbl.Y[sub_idx]) .>= 0), lnr, nothing; use_test_set=false)
                upper_dict, lower_dict = OCTHaGOn.trust_region_data(lnr, Symbol.(bbl.vars))


                push!(upper_dicts, upper_dict)
                push!(lower_dicts, lower_dict)
                push!(lnrs, lnr)
        #     catch
        #         println("Robust exploration $(i) failed")
        #         continue
        #     end
        end
        X = bbl.X
        leaves = [IAI.apply(lnr, X) for lnr in lnrs];
        preds = [IAI.predict(lnr, X) for lnr in lnrs];

        mn = abs.(0.5.-mean(hcat(preds...),dims=2))
        mask = (mn .<= 2/8)[:,1];

        sample_ids = (1:length(mask))[mask];
        sample_leaves = hcat([l[mask] for l in leaves]...);

        leaf_dict = Dict([tuple(sample_leaves[i,:]) => sample_ids[i] for i=1:size(sample_leaves,1)])
        
        X_all = DataFrame()

        for id in collect(values(leaf_dict))
            A = []
            b = []

            for i = 1:length(leaves)
                leaf = leaves[i][id]
                for (bb,aa) in upper_dicts[i][leaf]
                    push!(A, -aa)
                    push!(b, -bb)
                end
                for (bb,aa) in lower_dicts[i][leaf]
                    push!(A, aa)
                    push!(b, bb)
                end
            end

            A = hcat(A...)';

            x_start = collect(X[id,:]);

            N_samples = trunc(Int, size(X,1)*sampling_factor/length(leaf_dict))

            Xs = har_sample(A, b, x_start, N_samples);
            Xs = Matrix(hcat(Xs...)');
            Xs = DataFrame(Xs, string.(bbl.vars));

            X_all = append!(X_all, Xs)
        end

        return X_all
    # catch
    #     print("Couldn't do OCT sampling")
    #     println(stacktrace(catch_backtrace()))
    #     return DataFrame() 
    # end

end