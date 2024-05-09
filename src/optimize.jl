""" Very coarse solution for gm to check for feasibility and an initial starting point. """
function surveysolve(gm::GlobalModel)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    if !isempty(bbcs)
        for bbc in bbcs
            learn_constraint!(bbc)
            update_tree_constraints!(gm, bbc)
        end
    end
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        for bbr in bbrs
            learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6, minbucket = 2*length(bbr.vars))
            update_tree_constraints!(gm, bbr)
        end
    end
    optimize!(gm)
    return
end

"""
    function add_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    function add_relaxation_variables!(gm::GlobalModel, bbls::Array)

Populates relax_var attributes of all substructs. 
"""
function add_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if bbl.relax_var isa Real
        bbl.relax_var = @variable(gm.model)
        set_lower_bound(bbl.relax_var, 0)  
    end
    if !isempty(bbl.mi_constraints)
        clear_tree_constraints!(gm, bbl)
        @info "Clearing MI constraints from $(bbl.name) due to relaxation."
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            add_relaxation_variables!(gm, ll)
        end
    end
end

function add_relaxation_variables!(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        add_relaxation_variables!(gm, bbl)
    end
end

add_relaxation_variables!(gm::GlobalModel) = add_relaxation_variables!(gm, gm.bbls)

function clear_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if bbl.relax_var isa JuMP.VariableRef
        is_valid(gm.model, bbl.relax_var) || 
            throw(OCTHaGOnException("$(bbl.relax_var) is not a valid variable of  $(gm.model)."))
        delete(gm.model, bbl.relax_var)
        bbl.relax_var = 0
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            clear_relaxation_variables!(gm, ll)
        end
    end
end

function clear_relaxation_variables!(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        clear_relaxation_variables!(gm, bbl)
    end
end

clear_relaxation_variables!(gm::GlobalModel) = clear_relaxation_variables!(gm, gm.bbls)

"""
Includes relaxation variables in objective function 
"""
function relax_objective!(gm::GlobalModel, M::Real = 1e8)
    no = 0
    for bbl in gm.bbls
        no += M*bbl.relax_var
        for ll in bbl.lls
            no += M*ll.relax_var
        end
    end
    @objective(gm.model, Min, gm.objective + no)
    return
end

""" 
Removes relaxation variables from objective. 
"""
function tight_objective!(gm::GlobalModel)
    @objective(gm.model, Min, gm.objective)
end


"""
descend!(gm::GlobalModel; kwargs...)

Performs gradient descent on the last optimal solution in gm.solution_history.
In case of infeasibility, first projects the feasible point using the local
constraint gradients. 

# Optional kwargs:
max_iterations: maximum number of gradient steps or projections.
step_penalty: magnitude of penalty on step size during projections.
step_size: Size of 0-1 normalized Euclidian ball we can step. 
decay_rate: Exponential coefficient of step size reduction. 

"""
# function descend!(gm::GlobalModel; append_x0=true, kwargs...)
#     # Initialization
#     clear_tree_constraints!(gm) 


#     bbls = gm.bbls
#     vars = gm.vars
#     bounds = get_bounds(vars)

#     # Update descent algorithm parameters
#     if !isempty(kwargs) 
#         for item in kwargs
#             set_param(gm, item.first, item.second)
#         end
#     end

#     # Checking for a nonlinear objective
#     obj_bbl = nothing
#     if gm.objective isa VariableRef
#         obj_bbl = filter(x -> x.dependent_var == gm.objective, 
#                             [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor])
#         if isempty(obj_bbl)
#             obj_bbl = nothing
#         elseif length(obj_bbl) == 1
#             obj_bbl = obj_bbl[1]
#             println("Succsefully set obj bbl")
#             vars = [var for var in vars if var != obj_bbl.dependent_var]
#             bbls = [bbl for bbl in gm.bbls if bbl != obj_bbl]
#         elseif length(obj_bbl) > 1
#             throw(OCTHaGOnException("GlobaModel $(gm.name) has more than one BlackBoxRegressor" *
#                                " on the objective variable."))
#         end
#     end 

#     grad_shell = DataFrame(string.(vars) .=> [Float64[] for i=1:length(vars)])

#     # Add relaxation variables
#     add_relaxation_variables!(gm)

#     # Final checks
#     obj_gradient = copy(grad_shell)
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
#     if isempty(obj_gradient)
#         obj_gradient = DataFrame(string.(vars) .=> zeros(length(vars)))
#     end

#     # x0 initialization, and actual objective computation
#     x0 = make_feasible(gm, DataFrame(gm.solution_history[end, :]))
#     sol_vals = x0[:,string.(vars)]
#     if !isnothing(obj_bbl)
#         actual_cost = obj_bbl(x0)
#         x0[:, string(gm.objective)] = actual_cost
#         sol_vals = x0[:,string.(vars)]
#         feas_gap(gm, x0) # Checking feasibility gaps with updated objective

#         if append_x0
#             push!(gm.cost, actual_cost[1])
#             append!(gm.solution_history, x0) # Pushing last solution
#         end
#     end

#     # Descent direction, counting and book-keeping
#     d = @variable(gm.model, [1:length(vars)])
#     var_diff = []
#     max_diff = 0
#     for key in vars
#         push!(var_diff, maximum(bounds[key]) - minimum(bounds[key]))
#         if !isinf(var_diff[end])
#             max_diff = maximum([var_diff[end], max_diff])
#         end
#     end
#     if any(isinf.(var_diff))
#         @warn "Unbounded variables detected. " *
#               "PGD may fail to converge to a local minimum."
#         replace!(var_diff, Inf => max_diff)
#     end
#     replace!(var_diff, 0 => 1) # Sanitizing fixed values. 

#     ct = 0
#     d_improv = 1e5
#     prev_feas = false
#     feas = is_feasible(gm)
#     abstol = get_param(gm, :abstol)
#     tighttol = get_param(gm, :tighttol)
#     max_iterations = get_param(gm, :max_iterations)
#     step_penalty = get_param(gm, :step_penalty)
#     equality_penalty = get_param(gm, :equality_penalty)
#     step_size = get_param(gm, :step_size) 
#     decay_rate = get_param(gm, :decay_rate)
#     rho = get_param(gm, :momentum)
    
#     #v_old = x0
#     d_old = zeros(length(vars))
#     d_new = zeros(length(vars))
    
#     # WHILE LOOP
#     @info("Starting projected gradient descent...")
#     while (!feas || !prev_feas ||
#         (prev_feas && feas && d_improv >= get_param(gm, :abstol)) ||
#         ct == 0) && ct < max_iterations
#         prev_obj = gm.cost[end]
#         constrs = []
#         ct += 1
#         if max_iterations-ct < 2
#             rho = 0
#         end
#         if step_penalty == 0
#             step_penalty = maximum([1e3, 1e4*abs(prev_obj)])
#         end
#         if equality_penalty == 0
#             equality_penalty = 100*step_penalty
#         end
    
#         # Linear objective gradient and constraints
#         if !isnothing(obj_bbl)
#             update_gradients(obj_bbl, [length(obj_bbl.Y)])
#             obj_gradient = copy(grad_shell)
#             append!(obj_gradient, 
#                 DataFrame(obj_bbl.gradients[end,:]), cols = :subset) 
#             obj_gradient = coalesce.(obj_gradient, 0)
#             # Update objective constraints
#             push!(constrs, @constraint(gm.model, sum(Array(obj_gradient[end,:]) .* d) + 
#                               obj_bbl.dependent_var >= obj_bbl.Y[end]))
#         end

#         # Constraint evaluation
#         errors = 0
#         for bbl in bbls
#             update_gradients(bbl, [length(bbl.Y)])
#             constr_gradient = copy(grad_shell)
#             append!(constr_gradient, DataFrame(bbl.gradients[end,:]), cols = :subset)
#             constr_gradient = coalesce.(constr_gradient, 0)
#             if bbl isa BlackBoxClassifier
#                 error_diff = maximum([tighttol, abs(bbl.Y[end])])
#                 if bbl.equality
#                     push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
#                     push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] <= error_diff * bbl.relax_var))
#                     push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
#                     errors += bbl.relax_var^2
#                 elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
#                     push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
#                     push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
#                     errors += bbl.relax_var^2
#                 else # feasible to zero tolerance
#                     push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] >= 0))
#                 end
#             elseif bbl isa BlackBoxRegressor

#                 if isnothing(bbl.dependent_var)
#                     error_diff = maximum([tighttol, abs(bbl.optima[end] - bbl.actuals[end])])
#                     if bbl.equality
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + error_diff * bbl.relax_var >= bbl.Y[end]))
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) <= bbl.Y[end] + error_diff * bbl.relax_var))
#                         push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
#                         errors += bbl.relax_var^2
#                     elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
#                         push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
#                         errors += bbl.relax_var^2
#                     else # feasible to zero tolerance
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) >= bbl.Y[end]))
#                     end
#                 else
#                     error_diff = maximum([tighttol, abs(bbl.optima[end] - bbl.actuals[end])])
#                     if bbl.equality
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var + error_diff * bbl.relax_var >= bbl.Y[end]))
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var <= bbl.Y[end] + error_diff * bbl.relax_var))
#                         push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
#                         errors += bbl.relax_var^2
#                     elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
#                         push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
#                         errors += bbl.relax_var^2
#                     else # feasible to zero tolerance
#                         push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var >= bbl.Y[end]))
#                     end
#                 end
#             end
#             if !isempty(bbl.lls)
#                 n_lls = length(bbl.lls)
#                 update_gradients(bbl, collect(length(bbl.Y) - n_lls:length(bbl.Y)-1))
#                 for i = 1:n_lls
#                     ll = bbl.lls[i]
#                     constr_gradient = copy(grad_shell)
#                     append!(constr_gradient, # Changing the headers of the gradient 
#                         DataFrame(string.(ll.vars) .=> Array(bbl.gradients[end-n_lls-1+i,:])), cols = :subset)
#                     constr_gradient = coalesce.(constr_gradient, 0)
#                     Y_val = bbl.Y[end-n_lls-1+i]
#                     if bbl isa BlackBoxClassifier
#                         error_diff = maximum([tighttol, abs(Y_val)])
#                         if bbl.equality
#                             push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + Y_val + error_diff * ll.relax_var >= 0))
#                             push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + Y_val <= error_diff * ll.relax_var))
#                             push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
#                             errors += ll.relax_var^2
#                         elseif !is_feasible(ll) || (is_feasible(ll) && ll.feas_gap[end] <= 0)
#                             push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + Y_val + error_diff * ll.relax_var >= 0))
#                             push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
#                             errors += ll.relax_var^2
#                         else # feasible to zero tolerance
#                             push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + Y_val >= 0))
#                         end
#                     elseif bbl isa BlackBoxRegressor

#                         if isnothing(ll.dependent_var)
#                             error_diff = maximum([tighttol, abs(ll.optima[end] - ll.actuals[end])])
#                             if bbl.equality
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + error_diff * ll.relax_var >= Y_val))
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) <= Y_val + error_diff * ll.relax_var))
#                                 push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
#                                 errors += ll.relax_var^2
#                             elseif !is_feasible(ll) || (is_feasible(ll) && ll.feas_gap[end] <= 0)
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + error_diff * ll.relax_var >= Y_val))
#                                 push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
#                                 errors += ll.relax_var^2
#                             else # feasible to zero tolerance
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) >= Y_val))
#                             end
#                         else 
#                             error_diff = maximum([tighttol, abs(ll.optima[end] - ll.actuals[end])])
#                             if bbl.equality
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var + error_diff * ll.relax_var >= Y_val))
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var <= Y_val + error_diff * ll.relax_var))
#                                 push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
#                                 errors += ll.relax_var^2
#                             elseif !is_feasible(ll) || (is_feasible(ll) && ll.feas_gap[end] <= 0)
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var + error_diff * ll.relax_var >= Y_val))
#                                 push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
#                                 errors += ll.relax_var^2
#                             else # feasible to zero tolerance
#                                 push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var >= Y_val))
#                             end
#                         end
#                     end
#                 end
#             end
#         end

#         # Initializing descent direction
#         push!(constrs, @constraint(gm.model, d .== vars .- Array(sol_vals[end,:])))
#         if feas # Small gradient step
#             push!(constrs, @constraint(gm.model, sum((d ./ var_diff).^2) <= 
#                 step_size/exp(decay_rate*(ct-1)/max_iterations)))
#             @objective(gm.model, Min, 1)
#             @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d) + equality_penalty*errors)
#         else    # Projection step
#             @objective(gm.model, Min, 1)
#             @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d) + step_penalty*sum((d ./ var_diff).^2) + equality_penalty*errors)
#         end

#         # Optimizing the step, and finding next x0
#         optimize!(gm.model)
        
#         d_curr = getvalue.(d)
#         d_new = rho*d_old + d_curr
#         #d_new += random_point_in_sphere(length(d_new), norm(d_new)*0.1/exp(decay_rate*(ct-1)/max_iterations))
#         d_old = d_new
        
#         x0_old = collect(x0[1,string.(vars)])
#         x0_new = x0_old+d_new
        
#         x0 = make_feasible(gm, DataFrame(string.(vars) .=> x0_new))
        
#         if !isnothing(obj_bbl)
#             x0[!, string(gm.objective)] = obj_bbl(x0)
#         end
#         sol_vals = x0[:,string.(vars)]
#         feas_gap(gm, x0)
#         append!(gm.solution_history, x0)

#         # Measuring improvements
#         if !isnothing(obj_bbl)
#             push!(gm.cost, gm.solution_history[end, string(gm.objective)])
#         else
#             push!(gm.cost,  JuMP.getvalue(gm.objective))
#         end
#         d_improv = gm.cost[end-1] - gm.cost[end]

#         # Feasibility check
#         prev_feas = feas
#         feas = is_feasible(gm)

#         # Saving solution dict for JuMP-style recovery.
#         # TODO: do not regenerate soldict unless final solution. 
#         gm.soldict = Dict(key => JuMP.getvalue.(gm.model[key]) for (key, value) in gm.model.obj_dict)

#         # Delete gradient constraints (TODO: perhaps add constraints to avoid cycling?)
#         for con in constrs
#             delete(gm.model, con)
#         end
#     end

#     fincost = round(gm.cost[end], digits = -Int(round(log10(abstol))))
#     status = 0

#     # Termination criteria
#     if ct >= max_iterations
#         @info "Max iterations ($(ct)) reached."
#         if prev_feas && feas
#             @info "Solution is not converged to tolerance $(abstol)!" 
#             status = 1
#         elseif (feas && !prev_feas) && (abs(d_improv) <= 100*abstol)
#             @info "Solution is feasible and likely cycling, but the solution is close. Reduce step size and descend again. "
#             status = 2
#         elseif (!feas && prev_feas) && (abs(d_improv) <= 100*abstol)
#             @info "Solution is infeasible and likely cycling, but the solution is close. Reduce step size and descend again. "
#             status = 3
#         elseif ((feas && !prev_feas) || (prev_feas && !feas))
#             @info "Solution is likely cycling, with > $(abstol) changes in cost. Reduce step size and descend again."
#             status = 4
#         elseif !feas && !prev_feas
#             @info "Solution is infeasible to tolerance $(tighttol)."
#             status = 5
#         end
#         @info("Final cost is $(fincost).")
#     else 
#         @info("PGD converged in $(ct) iterations!")
#         @info("The optimal cost is $(fincost).")
#         status = 0
#     end

#     # Reverting objective, and deleting vars
#     relax_var = gm.relax_coeff ==0 ? 0 : gm.relax_coeff * gm.relax_var;
#     #relax_term = (gm.relax_coeff != 0) ? relax_coeff*relax_var : 0;
#     if gm.objective_sense == "Max"
#         @objective(gm.model, Max, gm.objective-relax_var)
#     else
#         @objective(gm.model, Min, gm.objective+relax_var)
#     end
#     delete(gm.model, d)
#     return status
# end

function descend!(gm::GlobalModel; append_x0=true,use_hessian=true, kwargs...)
    # Initialization
    clear_tree_constraints!(gm) 

    use_hessian = get_param(gm, :second_order_repair)

    bbls = gm.bbls
    vars = gm.vars
    bounds = get_bounds(vars)

    # Update descent algorithm parameters
    if !isempty(kwargs) 
        for item in kwargs
            set_param(gm, item.first, item.second)
        end
    end

    # Checking for a nonlinear objective
    obj_bbl = nothing
    if gm.objective isa VariableRef
        obj_bbl = filter(x -> x.dependent_var == gm.objective, 
                            [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor])
        if isempty(obj_bbl)
            obj_bbl = nothing
        elseif length(obj_bbl) == 1
            obj_bbl = obj_bbl[1]
            println("Succsefully set obj bbl")
            vars = [var for var in vars if var != obj_bbl.dependent_var]
            bbls = [bbl for bbl in gm.bbls if bbl != obj_bbl]
        elseif length(obj_bbl) > 1
            throw(OCTHaGOnException("GlobaModel $(gm.name) has more than one BlackBoxRegressor" *
                               " on the objective variable."))
        end
    end 

    grad_shell = DataFrame(string.(vars) .=> [Float64[] for i=1:length(vars)])

    # Add relaxation variables
    add_relaxation_variables!(gm)

    # Final checks
    obj_gradient = copy(grad_shell)
    if isnothing(obj_bbl)
        if gm.objective isa VariableRef
            append!(obj_gradient, DataFrame(string.(gm.objective) => 1), cols = :subset)
        elseif gm.objective isa JuMP.GenericAffExpr
            append!(obj_gradient, DataFrame(Dict(string(key) => value for (key,value) in gm.objective.terms)), cols = :subset)
        else
            @warn "Type of objective $(gm.objective) is unsupported."
        end
        obj_gradient = coalesce.(obj_gradient, 0)
    end
    if isempty(obj_gradient)
        obj_gradient = DataFrame(string.(vars) .=> zeros(length(vars)))
    end

    # x0 initialization, and actual objective computation
    x0 = make_feasible(gm, DataFrame(gm.solution_history[end, :]))
    sol_vals = x0[:,string.(vars)]
    if !isnothing(obj_bbl)
        actual_cost = obj_bbl(x0)
        x0[:, string(gm.objective)] = actual_cost
        sol_vals = x0[:,string.(vars)]
        feas_gap(gm, x0) # Checking feasibility gaps with updated objective

        if append_x0
            push!(gm.cost, actual_cost[1])
            append!(gm.solution_history, x0) # Pushing last solution
        end
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
    abstol = 1e-4 #get_param(gm, :abstol)
    tighttol = 1e-8 #get_param(gm, :tighttol)
    max_iterations = 100 #get_param(gm, :max_iterations)
    step_penalty = 10^8 #get_param(gm, :step_penalty)
    equality_penalty = 10^8 #get_param(gm, :equality_penalty)
    step_size = 1e-3 #get_param(gm, :step_size) 
    decay_rate = 2 #get_param(gm, :decay_rate)
    rho = get_param(gm, :momentum)
    
    obj_hess = zeros(length(vars),length(vars))
    
    #v_old = x0
    d_old = zeros(length(vars))
    d_new = zeros(length(vars))
    
    # WHILE LOOP
    @info("Starting projected gradient descent...")
    while (!feas || !prev_feas ||
        (prev_feas && feas && d_improv >= get_param(gm, :abstol)) ||
        ct == 0) && ct < max_iterations
        prev_obj = gm.cost[end]
        constrs = []
        ct += 1
        if max_iterations-ct < 2
            rho = 0
        end
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
            #print(sol_vals)
            if use_hessian
                try
                    obj_hess = obj_bbl.h(sol_vals[1,:])
                catch
                    print("Couldn't calculate obj hessian")
                end
            end
            #print("AFTER",sol_vals)
            append!(obj_gradient, 
                DataFrame(obj_bbl.gradients[end,:]), cols = :subset) 
            obj_gradient = coalesce.(obj_gradient, 0)
            # Update objective constraints
            push!(constrs, @constraint(gm.model, d'*obj_hess*d+sum(Array(obj_gradient[end,:]) .* d) + 
                              obj_bbl.dependent_var >= obj_bbl.Y[end]))
        end

        # Constraint evaluation
        errors = 0
        for bbl in bbls
            constr_hessian = zeros(length(vars), length(vars))
            
            if use_hessian
                try
                    constr_hessian = bbl.h(sol_vals[1,:])
                catch
                    print("Couldn't calculate constraint hessian")
                end
            end
            
            update_gradients(bbl, [length(bbl.Y)])
            constr_gradient = copy(grad_shell)
            append!(constr_gradient, DataFrame(bbl.gradients[end,:]), cols = :subset)
            constr_gradient = coalesce.(constr_gradient, 0)
            if bbl isa BlackBoxClassifier
                error_diff = maximum([tighttol, abs(bbl.Y[end])])
                if bbl.equality
                    push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                    push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] <= error_diff * bbl.relax_var))
                    push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                    errors += bbl.relax_var^2
                elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
                    push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                    push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                    errors += bbl.relax_var^2
                else # feasible to zero tolerance
                    push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] >= 0))
                end
            elseif bbl isa BlackBoxRegressor

                if isnothing(bbl.dependent_var)
                    error_diff = maximum([tighttol, abs(bbl.optima[end] - bbl.actuals[end])])
                    if bbl.equality
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d) + error_diff * bbl.relax_var >= bbl.Y[end]))
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d) <= bbl.Y[end] + error_diff * bbl.relax_var))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    else # feasible to zero tolerance
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d) >= bbl.Y[end]))
                    end
                else
                    error_diff = maximum([tighttol, abs(bbl.optima[end] - bbl.actuals[end])])
                    if bbl.equality
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var + error_diff * bbl.relax_var >= bbl.Y[end]))
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var <= bbl.Y[end] + error_diff * bbl.relax_var))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    elseif !is_feasible(bbl) || (is_feasible(bbl) && bbl.feas_gap[end] <= 0)
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] + error_diff * bbl.relax_var >= 0))
                        push!(constrs, @constraint(gm.model, bbl.relax_var <= 1))
                        errors += bbl.relax_var^2
                    else # feasible to zero tolerance
                        push!(constrs, @constraint(gm.model, d'*constr_hessian*d + sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var >= bbl.Y[end]))
                    end
                end
            end
            if !isempty(bbl.lls)
                n_lls = length(bbl.lls)
                update_gradients(bbl, collect(length(bbl.Y) - n_lls:length(bbl.Y)-1))
                for i = 1:n_lls
                    ll = bbl.lls[i]
                    constr_gradient = copy(grad_shell)
                    append!(constr_gradient, # Changing the headers of the gradient 
                        DataFrame(string.(ll.vars) .=> Array(bbl.gradients[end-n_lls-1+i,:])), cols = :subset)
                    constr_gradient = coalesce.(constr_gradient, 0)
                    Y_val = bbl.Y[end-n_lls-1+i]
                    if bbl isa BlackBoxClassifier
                        error_diff = maximum([tighttol, abs(Y_val)])
                        if bbl.equality
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + Y_val + error_diff * ll.relax_var >= 0))
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + Y_val <= error_diff * ll.relax_var))
                            push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
                            errors += ll.relax_var^2
                        elseif !is_feasible(ll) || (is_feasible(ll) && ll.feas_gap[end] <= 0)
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + Y_val + error_diff * ll.relax_var >= 0))
                            push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
                            errors += ll.relax_var^2
                        else # feasible to zero tolerance
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + Y_val >= 0))
                        end
                    elseif bbl isa BlackBoxRegressor

                        if isnothing(ll.dependent_var)
                            error_diff = maximum([tighttol, abs(ll.optima[end] - ll.actuals[end])])
                            if bbl.equality
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + error_diff * ll.relax_var >= Y_val))
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) <= Y_val + error_diff * ll.relax_var))
                                push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
                                errors += ll.relax_var^2
                            elseif !is_feasible(ll) || (is_feasible(ll) && ll.feas_gap[end] <= 0)
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + error_diff * ll.relax_var >= Y_val))
                                push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
                                errors += ll.relax_var^2
                            else # feasible to zero tolerance
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) >= Y_val))
                            end
                        else 
                            error_diff = maximum([tighttol, abs(ll.optima[end] - ll.actuals[end])])
                            if bbl.equality
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var + error_diff * ll.relax_var >= Y_val))
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var <= Y_val + error_diff * ll.relax_var))
                                push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
                                errors += ll.relax_var^2
                            elseif !is_feasible(ll) || (is_feasible(ll) && ll.feas_gap[end] <= 0)
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var + error_diff * ll.relax_var >= Y_val))
                                push!(constrs, @constraint(gm.model, ll.relax_var <= 1))
                                errors += ll.relax_var^2
                            else # feasible to zero tolerance
                                push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var >= Y_val))
                            end
                        end
                    end
                end
            end
        end

        # Initializing descent direction
        push!(constrs, @constraint(gm.model, d .== vars .- Array(sol_vals[end,:])))
        if feas # Small gradient step
            push!(constrs, @constraint(gm.model, sum((d ./ var_diff).^2) <= 
                step_size/exp(decay_rate*(ct-1)/max_iterations)))
            @objective(gm.model, Min, 1)
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d)+d'*obj_hess*d + equality_penalty*errors)
        else    # Projection step
            @objective(gm.model, Min, 1)
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d)+d'*obj_hess*d + step_penalty*sum((d ./ var_diff).^2) + equality_penalty*errors)
        end

        # Optimizing the step, and finding next x0
        optimize!(gm.model)
        
        d_curr = getvalue.(d)
        d_new = rho*d_old + d_curr
        #d_new += random_point_in_sphere(length(d_new), norm(d_new)*0.1/exp(decay_rate*(ct-1)/max_iterations))
        d_old = d_new
        
        x0_old = collect(x0[1,string.(vars)])
        x0_new = x0_old+d_new
        
        x0 = make_feasible(gm, DataFrame(string.(vars) .=> x0_new))
        
        if !isnothing(obj_bbl)
            x0[!, string(gm.objective)] = obj_bbl(x0)
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

        try
            # Saving solution dict for JuMP-style recovery.
            # TODO: do not regenerate soldict unless final solution. 
            gm.soldict = Dict(key => JuMP.getvalue.(gm.model[key]) for (key, value) in gm.model.obj_dict)

            # Delete gradient constraints (TODO: perhaps add constraints to avoid cycling?)
            for con in constrs
                delete(gm.model, con)
            end
        catch
            # Delete gradient constraints (TODO: perhaps add constraints to avoid cycling?)
            for con in constrs
                delete(gm.model, con)
            end
            throw(error("Infeasible descend"))
        end
    end

    status = 0
    fincost = round(gm.cost[end], digits = -Int(round(log10(abstol))))
    # Termination criteria
    if ct >= max_iterations
        @info "Max iterations ($(ct)) reached."
        if prev_feas && feas
            @info "Solution is not converged to tolerance $(abstol)!" 
            status = 1
        elseif (feas && !prev_feas) && (abs(d_improv) <= 100*abstol)
            @info "Solution is feasible and likely cycling, but the solution is close. Reduce step size and descend again. "
            status = 2
        elseif (!feas && prev_feas) && (abs(d_improv) <= 100*abstol)
            @info "Solution is infeasible and likely cycling, but the solution is close. Reduce step size and descend again. "
            status = 3
        elseif ((feas && !prev_feas) || (prev_feas && !feas))
            @info "Solution is likely cycling, with > $(abstol) changes in cost. Reduce step size and descend again."
            status = 4
        elseif !feas && !prev_feas
            @info "Solution is infeasible to tolerance $(tighttol)."
            status = 5
        end
        @info("Final cost is $(fincost).")
    else 
        @info("PGD converged in $(ct) iterations!")
        @info("The optimal cost is $(fincost).")
        status = 0
    end

    # Reverting objective, and deleting vars
    relax_var = gm.relax_coeff ==0 ? 0 : gm.relax_coeff * gm.relax_var;
    #relax_term = (gm.relax_coeff != 0) ? relax_coeff*relax_var : 0;
    if gm.objective_sense == "Max"
        @objective(gm.model, Max, gm.objective-relax_var)
    else
        @objective(gm.model, Min, gm.objective+relax_var)
    end
    delete(gm.model, d)
    return status
end

""" Complete solution procedure for GlobalModel. """
function globalsolve!(gm::GlobalModel; repair=true, sampling_methods=["boundary", "lh", "knn", "derivative", "oct"])
    
    clear_tree_constraints!(gm)

    @info "GlobalModel " * gm.name * " solution in progress..."

    relax_var = gm.relax_coeff ==0 ? 0 : gm.relax_coeff * gm.relax_var;
    if gm.objective_sense == "Max"
        @objective(gm.model, Max, gm.objective-relax_var)
    else
        @objective(gm.model, Min, gm.objective+relax_var)
    end

    
    for bbl in gm.bbls
        if !is_sampled(bbl)
            uniform_sample_and_eval!(bbl; sampling_methods=sampling_methods, gm=gm)
        end
    end
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)
    @info "Training OptimalTreeLearners..."
    for bbl in gm.bbls
        if isempty(bbl.learners)
            learn_constraint!(bbl)
        end
        if isempty(bbl.mi_constraints) 
            add_tree_constraints!(gm, bbl)
        end
    end

    @info "Solving MIP..."
    # print(gm.model)

    set_optimizer_attribute(gm.model, "TimeLimit", 60)
    optimize!(gm) 

    gm.relax_epsilon = JuMP.value.(gm.relax_var)[1]

    if repair    
        try
            set_optimizer_attribute(gm.model, "TimeLimit", 30)
            println("Starting descend")
            set_param(gm, :step_penalty, 0.)
            set_param(gm, :equality_penalty, 0.)
            status = descend!(gm)
            if status >= 2
                set_param(gm, :step_penalty, 10^8)
                set_param(gm, :equality_penalty, 10^8)
                descend!(gm)
            end
            
        catch
            @warn("Descend failed") 
        end
         
        # final_repair!(gm)
        try
            final_sample_repair!(gm)
        catch
            @warn("Final repair failed") 
        end
    end

end 

function globalsolve_and_time!(m::Union{JuMP.Model, GlobalModel})
    t1 = time()
    if m isa JuMP.Model
        optimize!(m);
    else
        globalsolve!(m)
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
end

function final_repair!(gm::GlobalModel)

    try
        df2 = copy(gm.solution_history)
        df2[!, "feasible"] .= [false]

        for i=1:size(df2,1)
            feas_gap(gm, df2[i:i, [c for c in names(df2) if c != "feasible"]])
            df2[i, "feasible"] = all(abs.([bbl.feas_gap[end] for bbl in gm.bbls]).<=1e-3)
        end

        df_feas = df2[df2[!,"feasible"],:]
        best_sol = sort(df_feas, :objvar)[1:1,:]
        append!(gm.solution_history, best_sol, cols=:intersect)
        push!(gm.cost, gm.solution_history[end, string(gm.objective)])
        print("Solution repaired")
        #@warning("Solution repaired")
    catch
        #@info("Did not repair solution") 
    end
end