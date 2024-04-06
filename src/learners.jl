# @with_kw mutable struct SVM_Classifier
#     β0::Real
#     β::Vector{Float64}
# end

# @with_kw mutable struct SVM_Regressor
#     β0::Real
#     β::Vector{Float64}
# end

""" Returns the baseline OptimalTreeLearners, with all valid training args. """
function base_classifier()
    IAI.OptimalTreeClassifier(
        hyperplane_config = (sparsity = :all,),
        random_seed = 1,
        max_depth = 5, 
        cp = 1e-6,
        minbucket = 0.01,
        fast_num_support_restarts = 5,
        localsearch = true,
        ls_num_tree_restarts = 10, 
        ls_num_hyper_restarts =  5)
end

function base_regressor(;kwargs...)
    IAI.OptimalTreeRegressor(
        hyperplane_config = (sparsity = :all,),
        regression_sparsity = :all,
        regression_weighted_betas = true,
        random_seed = 1,
        max_depth = 5,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        regression_lambda = 1e-5,
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end

function base_rf_regressor()
    IAI.Heuristics.RandomForestRegressor(
        num_trees = 20,
        hyperplane_config = (sparsity = :all,),
        regression_sparsity = 0,
        regression_weighted_betas = true,
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        regression_lambda = 1e-5,
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end

function base_rf_classifier()
    IAI.Heuristics.RandomForestClassifier(
        num_trees = 20,
        hyperplane_config = (sparsity = :all,),
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end

function base_cart_classifier(;dependent_var=nothing)
    IAI.OptimalTreeClassifier(
       # hyperplane_config = (sparsity = :none,),
        random_seed = 1,
        max_depth = 5, 
        cp = 1e-6,
        minbucket = 0.01,
        fast_num_support_restarts = 5,
        localsearch = false,
        ls_num_tree_restarts = 10, 
        ls_num_hyper_restarts =  5)
end

function base_cart_regressor(;dependent_var=nothing)
    IAI.OptimalTreeRegressor(
        # hyperplane_config = (sparsity = :none,),
        regression_sparsity = :all,
        regression_weighted_betas = true,
        random_seed = 1,
        max_depth = 5,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        localsearch = false,
        regression_lambda = 1e-5,
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end


# Dictionary that registers the
# generators for the different models
LEARNER_DICT = Dict(
    "classification" => Dict(
        "OCT" => base_classifier,
        "RF" => base_rf_classifier,
        "CART" => base_cart_classifier,
        "SVM" => SVM_Classifier,
        "GBM" => GBM_Classifier,
        "MLP" => MLP_Classifier
    ),
    "regression" => Dict(
        "OCT" => base_regressor,
        "RF" => base_rf_regressor,
        "CART" => base_cart_regressor,
        "SVM" => SVM_Regressor,
        "GBM" => GBM_Regressor,
        "MLP" => MLP_Regressor
    )
)