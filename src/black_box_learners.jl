

function Base.show(io::IO, lc::LinkedClassifier)
    println(io, "LinkedClassifier with: ")
    println(io, "variables: $(lc.vars) ") # TODO: improve printing
end



function Base.show(io::IO, lr::LinkedRegressor)
    println(io, "LinkedRegressor with: ")
    println(io, "variables: $(lr.vars) ") # TODO: improve printing
    println(io, "and dependent variable: $(lr.dependent_var)")
end


function Base.show(io::IO, bbr::BlackBoxRegressor)
    if bbr.equality
        println(io, "BlackBoxRegressor EQUALITY " * bbr.name * " with $(length(bbr.vars)) independent variables, ")
    else
        println(io, "BlackBoxRegressor inequality " * bbr.name * " with $(length(bbr.vars)) independent variables, ")
    end
    println(io, "and dependent variable $(bbr.dependent_var).")
    println(io, "Sampled $(length(bbr.Y)) times, and has $(length(bbr.learners)) trained ORTs.")
    if !isempty(bbr.lls)
        println(io, "Has $(length(bbr.lls)) linked constraints.")
    end
end



function Base.show(io::IO, bbc::BlackBoxClassifier)
    if bbc.equality
        println(io, "BlackBoxClassifier EQUALITY " * bbc.name * " with $(length(bbc.vars)) variables: ")
    else
        println(io, "BlackBoxClassifier inequality " * bbc.name * " with $(length(bbc.vars)) variables: ")
    end
    println(io, "Sampled $(length(bbc.Y)) times, and has $(length(bbc.learners)) trained OCTs.")
    if !isempty(bbc.lls)
        println(io, "Has $(length(bbc.lls)) linked constraints.")
    end
    if get_param(bbc, :ignore_feasibility)
        if get_param(bbc, :ignore_accuracy)
            println(io, "Ignores training accuracy and data_feasibility thresholds.")
        else
            println(io, "Ignores data feasibility thresholds.")
        end
    else
        if get_param(bbc, :ignore_accuracy)
            println("Ignores training accuracy thresholds.")
        end
    end
end

""" BBL type is for function definitions! """

set_param(bbl::BlackBoxLearner, key::Symbol, val) = set_param(bbl.params, key, val)
set_param(bbls::Array{BlackBoxLearner}, key::Symbol, val) = foreach(bbl -> set_param(bbl, key, val), bbls)
get_param(bbl::BlackBoxLearner, key::Symbol) = get_param(bbl.params, key)
get_param(bbls::Array{BlackBoxLearner}, key::Symbol) = [get_param(bbl, key) for bbl in bbls]

"""
    add_data!(bbc::BlackBoxClassifier, X::DataFrame, Y::Array)

Adds data to BlackBoxClassifier.
"""
function add_data!(bbc::BlackBoxClassifier, X::DataFrame, Y::Array)
    @assert length(Y) == size(X, 1)
    if size(bbc.X,1) == 0
        bbc.feas_ratio = sum(Y .>= 0)/length(Y)
    else
        bbc.feas_ratio = (bbc.feas_ratio*size(bbc.X,1) + sum(Y .>= 0))/(size(bbc.X, 1) + length(Y))
    end
    if isnothing(bbc.gradients) && get_param(bbc, :gradients) # TODO: improve the gradient DF init. 
        bbc.gradients = DataFrame(string.(bbc.vars) .=> [Union{Missing, Float64}[] for i=1:length(bbc.vars)]) 
        bbc.curvatures = Union{Missing, Float64}[]
    end
    if !isnothing(bbc.gradients)
        append!(bbc.gradients, DataFrame(string.(bbc.vars) .=> 
            [missings(size(X, 1)) for i=1: length(bbc.vars)]), cols=:intersect)
        append!(bbc.curvatures, missings(size(X,1)))   
    end    
    append!(bbc.X, X, cols=:intersect)
    append!(bbc.Y, Y)
    return
end

"""
    add_data!(bbr::BlackBoxRegressor, X::DataFrame, Y::Array)

Adds data to BlackBoxRegressor.
"""
function add_data!(bbr::BlackBoxRegressor, X::DataFrame, Y::Array)
    @assert length(Y) == size(X, 1)
    infeas_idxs = findall(x -> isinf(x), Y)
    # length(infeas_idxs) >= 0.5*length(Y) &&
    # throw(OCTHaGOnException("A majority of samples on BBR $(bbr.name) evaluated as infeasible. " * 
    #     "Please check your expressions. "))
    if isnothing(bbr.gradients) && get_param(bbr, :gradients) 
        bbr.gradients = DataFrame(string.(bbr.vars) .=> [Union{Missing, Float64}[] for i=1:length(bbr.vars)]) 
        bbr.curvatures = Union{Missing, Float64}[]
    end 
    if !isempty(infeas_idxs)
        append!(bbr.infeas_X, X[infeas_idxs, :], cols=:intersect)
        clean_X = delete!(X, infeas_idxs)
        if !isnothing(bbr.gradients)
            append!(bbr.gradients, DataFrame(string.(bbr.vars) .=> 
                        [missings(size(clean_X, 1)) for i=1:length(bbr.vars)]), cols=:intersect)
            append!(bbr.curvatures, missings(size(clean_X,1)))
        end
        append!(bbr.X, clean_X, cols=:intersect)
        append!(bbr.Y, deleteat!(Y, infeas_idxs))
    else
        if !isnothing(bbr.gradients)
            append!(bbr.gradients, DataFrame(string.(bbr.vars) .=> 
                        [missings(size(X, 1)) for i=1:length(bbr.vars)]), cols=:intersect)
            append!(bbr.curvatures, missings(size(X,1)))
        end
        append!(bbr.X, X, cols=:intersect)
        append!(bbr.Y, Y)
    end
    return
end

"""
    evaluate(bbl::BlackBoxLearner, data::Union{Dict, DataFrame})

Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform.
"""
function evaluate(bbl::BlackBoxLearner, data::Union{Dict, DataFrame})
    @assert !isnothing(bbl.constraint)
    clean_data = data_to_DataFrame(data);
    if isnothing(bbl.f)
        constr_obj = constraint_object(bbl.constraint)
        rhs_const = get_constant(constr_obj.set)
        vals = [JuMP.value(constr_obj.func, i -> clean_data[!,string(i)][j]) for j=1:size(clean_data,1)]
        if any(isinf.(vals)) || any(isnan.(vals))
            throw(OCTHaGOnException(string("Constraint ", bbl.constraint, " returned an infinite or NaN value.",
                                      "Please check your variable definitions in bbl ", bbl, " . ")))
        end
        if isnothing(rhs_const)
            vals = [-1*distance_to_set(vals[i], constr_obj.set) for i=1:length(vals)]
        else
            coeff=1
            if constr_obj.set isa MOI.LessThan
                coeff=-1
            end
            vals = coeff.*(vals .- rhs_const)
        end
        length(vals) == 1 && return vals[1]
        return vals
    else
        arrs = deconstruct(clean_data, bbl.vars, bbl.expr_vars, bbl.varmap)
        vals = []
        for arr in arrs
            try
                push!(vals, Base.invokelatest(bbl.f, (arr...)))
            catch DomainError
                #stacktrace(catch_backtrace())
                push!(vals, -Inf)
            end
        end
        return vals
    end
end

"""
    evaluate_gradient(bbl::BlackBoxLearner, data::Union{Array, DataFrame})

Evaluates gradient of function through ForwardDiff. 
TODO: speed-ups!. 
"""
# function evaluate_gradient(bbl::BlackBoxLearner, data::DataFrame)
#     @assert get_param(bbl, :gradients)
#     @assert !isnothing(bbl.constraint)
#     @assert(size(data, 2) == length(bbl.vars))
#     if bbl.constraint isa JuMP.ConstraintRef
#         return DataFrame(hcat([bbl.g(Array(row)) 
#                 for row in eachrow(data[:, string.(bbl.vars)])]...)', string.(bbl.vars))
#     elseif bbl.constraint isa Expr
        
#         bounds = get_bounds(bbl)
#         bounds = Dict(string(key)=>value for (key, value) in bounds)
#         lbs = [bounds[v][1] for v in string.(bbl.vars)]
#         ubs = [bounds[v][2] for v in string.(bbl.vars)]
#         arrs = deconstruct(data, bbl.vars, bbl.expr_vars, bbl.varmap)
#         grads = hcat([bbl.g(Float64[flat(arr)...]) 
#                             for arr in arrs]...)'
        
#         grad_rows = []
#         for arr in arrs
#             row = Float64[flat(arr)...]
#             grad_row = []
#             for j=1:3
#                 grad_row = bbl.g(Array(row))
#                 inf_vars_ids = (1:length(grad_row))[(grad_row .== Inf) .| isnan.(grad_row)]
#                 if length(inf_vars_ids) ==0
#                     break
#                 end
#                 signs = 2*(row-lbs .<=ubs-row ).-1
#                 perts = []

#                 for i=1:length(lbs)
#                     pert = signs[i] == 1 ? (ubs[i]-row[i])*1e-2 : -(row[i]-lbs[i])*1e-2;
#                     if i ∉ inf_vars_ids
#                         pert = 0
#                     end
#                     push!(perts, pert)
#                 end
#                 # Perturb gradients if they are infinite
#                 row += perts
#             end
#             push!(grad_rows, grad_row)
            
#         end
#         grads = hcat(grad_rows...)'
        
#         return DataFrame(grads[:, bbl.datamap], string.(bbl.vars))
#     end
# end
function evaluate_gradient(bbl::BlackBoxLearner, data::DataFrame)
    @assert get_param(bbl, :gradients)
    @assert !isnothing(bbl.constraint)
    @assert(size(data, 2) == length(bbl.vars))
    if bbl.constraint isa JuMP.ConstraintRef
        return DataFrame(hcat([bbl.g(Array(row)) 
                for row in eachrow(data[:, string.(bbl.vars)])]...)', string.(bbl.vars))
    elseif bbl.constraint isa Expr
        arrs = deconstruct(data, bbl.vars, bbl.expr_vars, bbl.varmap)
        grads = hcat([bbl.g(Float64[flat(arr)...]) 
                            for arr in arrs]...)'
        return DataFrame(grads[:, bbl.datamap], string.(bbl.vars))
    end
end

"""
    update_gradients(bbl::BlackBoxLearner, idxs = collect(1:size(bbl.X,1)))

Updates gradient information of selected points. 
"""
function update_gradients(bbl::BlackBoxLearner, idxs::Array = collect(1:size(bbl.X,1)))
    @assert get_param(bbl, :gradients)
    isempty(idxs) && return
    empties = idxs[findall(idx -> any(ismissing.(values(bbl.gradients[idx,:]))), idxs)]
    isempty(empties) && return
    bbl.gradients[empties, :] = evaluate_gradient(bbl, bbl.X[empties, :])
    return
end    

"""
    function (bbl::BlackBoxLearner)(x::Union{DataFrame,Dict,DataFrameRow})

Makes the BBC or BBR callable as a function.
"""
function (bbl::BlackBoxLearner)(X::Union{DataFrame,Dict,DataFrameRow})
    return evaluate(bbl, X)
end

""" Evaluates BlackBoxLearner at all X and stores the resulting data. """
function eval!(bbl::BlackBoxLearner, X::DataFrame)
    Y = bbl(X);
    add_data!(bbl, X, Y)
end

""" 
    clear_tree_data!(bbc::BlackBoxClassifier)
    clear_tree_data!(bbr::BlackBoxRegressor)

Deletes tree data associated with object. 
"""
function clear_tree_data!(bbc::BlackBoxClassifier)
    bbc.learners = [];
    bbc.learner_kwargs = []                            
    bbc.accuracies = []
end

function clear_tree_data!(bbr::BlackBoxRegressor)
    bbr.learners = [];
    bbr.learner_kwargs = []   
    bbr.thresholds = []                         
    bbr.ul_data = Dict[]            
end

""" 
    clear_data!(bbc::BlackBoxClassifier)
    clear_data!(bbr::BlackBoxRegressor)    

Deletes all data (NOT constraints) associated with object. 
Please clear constraints using the clear_tree_constraints! function. 
"""
function clear_data!(bbc::BlackBoxClassifier)
    bbc.X = DataFrame(string.(bbc.vars) .=> [Float64[] for i=1:length(bbc.vars)])
    bbc.Y = [];
    bbc.gradients = nothing
    bbc.curvatures = nothing
    bbc.feas_ratio = 0
    clear_tree_data!(bbc)
end

function clear_data!(bbr::BlackBoxRegressor)
    bbr.X = DataFrame(string.(bbr.vars) .=> [Float64[] for i=1:length(bbr.vars)])
    bbr.Y = []
    bbr.gradients = nothing
    bbr.curvatures = nothing
    bbr.infeas_X = DataFrame(string.(bbr.vars) .=> [Float64[] for i=1:length(bbr.vars)]);
    clear_tree_data!(bbr)
                       
end

""" Helper function to find which binary leaf variables are one. """
function active_leaves(bbl::Union{BlackBoxLearner, LinkedLearner})
    leaf_in = []
    for (leaf, var_tuple) in bbl.leaf_variables
        if isapprox(getvalue(var_tuple[1]), 1; atol=1e-5)
            push!(leaf_in, leaf)
        end
    end
    bbl.active_leaves = leaf_in
end

""" 
    all_mi_constraints(bbl::BlackBoxLearner)

Returns all JuMP.ConstraintRefs associated with BBL. 
"""
function all_mi_constraints(bbl::BlackBoxLearner, hypertype::Union{String, Nothing} = nothing)
    all_constraints = []
    if hypertype == nothing
        for (leaf, constraints) in bbl.mi_constraints
            push!(all_constraints, constraints...)
        end
    elseif hypertype in valid_lowers
        for (leaf, constraints) in bbl.mi_constraints
            if leaf >= 0 
                push!(all_constraints, constraints...)
            end
        end
    end
    if hypertype in valid_uppers
        for (leaf, constraints) in bbl.mi_constraints
            if leaf <= 0 
                push!(all_constraints, constraints...)
            end
        end
    end
    return all_constraints
end

"""
    active_lower_tree(bbr::BlackBoxRegressor)

Returns the index of currently active lower bounding tree of BBR. 
"""
function active_lower_tree(bbr::BlackBoxRegressor)
    if length(bbr.active_trees) > 2
        throw(OCTHaGOnException("Regressor $(bbr.name) has too many active trees. There is a bug in the update. "))
    elseif length(bbr.active_trees) == 1
        if collect(values(bbr.active_trees))[1].first in valid_lowers
            return collect(keys(bbr.active_trees))[1]
        else
            return # no active lower trees
        end
    elseif length(bbr.active_trees) == 2
        tree_keys = collect(keys(bbr.active_trees))
        tree_hypertypes = collect(values(bbr.active_trees))
        if tree_hypertypes[1].first == "lower"
            return tree_keys[1]
        elseif tree_hypertypes[2].first == "lower"
            return tree_keys[2]
        elseif tree_hypertypes[1].first in valid_lowers
            return tree_keys[1]
        else
            return tree_keys[2]
        end
    else
        return # no active trees
    end
end

"""
    active_upper_tree(bbr::BlackBoxRegressor)

Returns the index of currently active active_upper_tree bounding tree of BBR. 
"""
function active_upper_tree(bbr::BlackBoxRegressor)
    if length(bbr.active_trees) == 1
        if collect(values(bbr.active_trees))[1].first in valid_uppers
            return collect(keys(bbr.active_trees))[1]
        else
            return # no active upper trees
        end
    elseif length(bbr.active_trees) == 2
        tree_keys = collect(keys(bbr.active_trees))
        tree_hypertypes = collect(values(bbr.active_trees))
        if tree_hypertypes[1].first == "upper"
            return tree_keys[1]
        elseif tree_hypertypes[2].first == "upper"
            return tree_keys[2]
        elseif tree_hypertypes[1].first in valid_uppers
            return tree_keys[1]
        else
            return tree_keys[2]
        end
    else
        return # no active trees
    end
end

""" 
    update_local_convexity(bbl::BlackBoxLearner)

Checks proportion of "neighbor convex" sample points of BBL. 
"""
function update_local_convexity(bbl::BlackBoxLearner)
    classify_curvature(bbl)
    bbl.local_convexity = sum(bbl.curvatures .== 1) / size(bbl.X, 1)
    return
end

""" 
    update_vexity(bbl::BlackBoxLearner)

Checks whether a function is perhaps locally or globally convex.
Threshold sets the border of being considered for convex regression. 
"""
function update_vexity(bbl::BlackBoxLearner, t::Int64 = 5)
    update_local_convexity(bbl)
    thresh = 1e-10 * (bbl.max_Y - bbl.min_Y)
    if bbl.local_convexity == 1.0
        if bbl isa BlackBoxRegressor
            # Checking against quasi_convexity with 5 random points
            cvx = true
            test_idxs = Int64.(ceil.(rand(t) .* size(bbl.X, 1)))
            diffs = [[Array(bbl.X[j, :]) - Array(bbl.X[i, :]) for i in test_idxs] for j in test_idxs]
            for i=1:t, j=1:t
                if i != j && !(bbl.Y[test_idxs[j]] >= bbl.Y[test_idxs[i]] - 
                                sum(Array(bbl.gradients[test_idxs[i],:]) .* diffs[i][j]) - thresh)
                    cvx = false
                    @info "Points $(test_idxs[j]) and $(test_idxs[i]) broke the convexity of $(bbl.name)."
                    break
                end
            end
            if cvx
                bbl.convex = true
            end
        else
            # Checking against quasi_convexity with 5 random points
            cvx = true
            test_idxs = Int64.(ceil.(rand(t) .* size(bbl.X, 1)))
            diffs = [[Array(bbl.X[j, :]) - Array(bbl.X[i, :]) for i in test_idxs] for j in test_idxs]
            for i=1:t, j=1:t
                if i != j && !(bbl.Y[test_idxs[i]] >= bbl.Y[test_idxs[j]] + 
                                sum(Array(bbl.gradients[test_idxs[i],:]) .* diffs[i][j]) - thresh)
                    cvx = false
                    @info "Points $(test_idxs[i]) and $(test_idxs[j]) broke the convexity of $(bbl.name)."
                    break
                end
            end
            if cvx
                bbl.convex = true
            end
        end
    end
    return
end

"""
    update_leaf_vexity(bbl::BlackBoxLearner)

Finds the local convexity of leaves (bbl.vexity) of the active lower bounding tree of BBL. 
"""
function update_leaf_vexity(bbl::BlackBoxLearner)
    if bbl.convex
        bbl.vexity[1] = (size(bbl.X, 1), 1.0) # We only have a "root". 
        return 
    end
    lnr = bbl.learners[end] # last tree is always the active tree...
    leaf_idxs = IAI.apply(lnr, bbl.X)
    all_leaves = find_leaves(lnr)
    leaf_vexity = Dict()
    if lnr isa BlackBoxClassifier
        all_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
    end
    if any(ismissing.(bbl.curvatures))
        classify_curvature(bbl)
    end
    for leaf in all_leaves
        in_leaf_idxs = findall(x -> x == leaf, leaf_idxs)
        leaf_vexity[leaf] = (length(in_leaf_idxs), sum(bbl.curvatures[in_leaf_idxs] .> 0) / length(in_leaf_idxs))
    end
    bbl.vexity = leaf_vexity
    return
end