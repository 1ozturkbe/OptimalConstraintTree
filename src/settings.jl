""" Returns default BlackBoxRegressor settings for approximation."""
function bbr_defaults(n_vars::Int64 = 10; sample_coeff = 200)
    Dict(:n_samples => Int(ceil(sample_coeff*sqrt(n_vars))), # (0 if no sampling fn)
        :gradients => true,                                  # TODO: add option whether or not to use gradients
        :reloaded => false)                                  # Whether learners are reloaded
end

""" Returns default BlackBoxClassifier settings for approximation."""
function bbc_defaults(n_vars::Int64 = 10; sample_coeff = 200)
    Dict(:threshold_accuracy => 0.95,                       # Minimum tree accuracy
        :threshold_feasibility => 0.15,                     # Minimum feasibility ratio
        :ignore_feasibility => false,                       # Whether we should ignore feasibility checks
        :ignore_accuracy => false,                          # Whether we should ignore accuracy checks 
        :n_samples => Int(ceil(sample_coeff*sqrt(n_vars))), # (0 if no sampling fn)
        :gradients => true,
        :reloaded => false)                                 # Whether learners are reloaded  
end

"""
Returns default GlobalModel settings for approximation and optimization.
"""
function gm_defaults()
    Dict(:ignore_feasibility => false,
         :ignore_accuracy => false,
         :lh_iterations => 5,
         :sample_coeff => 200,
         :sample_density => 1e-5, # sets maximum sample density in Euclidian distance
         :abstol => 1e-4,
         :tighttol => 1e-8, # sets equality constraint tolerance. 
         :convex_constrs => true,
         # Params for gradient descent
         :max_iterations => 100,
         :step_size => 1e-3,
         :step_penalty => 0.,
         :decay_rate => 2.,
         :equality_penalty => 0.,
         :momentum => 0.,
         :second_order_repair => false,
         :oct_sampling => false,
         # Params for robust optimization
         :ro_factor => 0.
    )
end

""" Sets parameters within Dict. """
function set_param(gm::Dict, key::Symbol, val, checks = true)
    if haskey(gm, key) && val isa typeof(gm[key])
        gm[key] = val
        return
    elseif checks
        throw(OCTHaGOnException("Parameter with key " * string(key) * " is invalid."))
    else 
        return
    end
end

""" gets parameters within Dict. """
function get_param(gm::Dict, key::Symbol)
    if haskey(gm, key)
        return gm[key]
    else
        throw(OCTHaGOnException("Parameter with key " * string(key) * " is invalid."))
    end
end