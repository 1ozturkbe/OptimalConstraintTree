module OCTHaGOn
    using Combinatorics
    using CPLEX
    using DataFrames
    using ForwardDiff
    using JuMP
    using LatinHypercubeSampling
    using LinearAlgebra
    using MacroTools
    using Main.IAI
    using MathOptInterface
    using MathOptSetDistances
    using Missings
    using NearestNeighbors
    using Parameters
    using ProgressMeter
    using Random
    using Clustering
    using Knet
    using Statistics
    using SHA
    using Distributions
    using StatsBase
    using DecisionTree
    using ScikitLearn
    using Gurobi

    const PROJECT_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = PROJECT_ROOT * "\\data\\"
    const BARON_DIR = PROJECT_ROOT * "\\data\\baron\\"
    const GAMS_DIR = PROJECT_ROOT * "\\data\\gams\\"  
    const SOL_DIR = PROJECT_ROOT * "\\data\\solutions\\"
    const TREE_DIR = PROJECT_ROOT * "\\data\\trees\\"
    # const CPLEX_SILENT = with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0, CPX_PARAM_TILIM = 300)
    const CPLEX_SILENT = with_optimizer(Gurobi.Optimizer, TimeLimit = 60, NonConvex = 2, OutputFlag = 0)# , FeasibilityTol=1e-3, OptimalityTol=1e-3

    const valid_lowers = ["reg", "lower", "upperlower"]
    const valid_uppers = ["reg", "upper", "upperlower"]
    const valid_singles = ["reg", "upperlower"]
    const valid_pairs =  ["lower" => "upper",
                          "upper" => "lower",
                          "reg" => "upper", # reg with some threshold.second
                          "upper" => "reg"] # reg with some threshold.second

    include("structs.jl")

    include("small_scripts.jl")

    include("on_jump.jl")

    include("util/train_test_split.jl")

    
    include("learners/abstract.jl")
    include("learners/svm.jl")
    include("learners/gbm.jl")
    include("learners/mlp.jl")
    include("learners.jl")

    #include("learners/iai.jl")

    include("exceptions.jl")

    include("settings.jl")
    
    include("black_box_learners.jl")

    include("root_finding.jl")

    include("iai_wrappers.jl")

    include("global_model.jl")

    include("robust.jl")
    
    include("data_driven.jl")

    include("sample.jl")

    include("constraintify.jl")

    include("fit.jl")

    include("refine.jl")

    include("optimize.jl")

    include("repair.jl")

    include("plot.jl")

    include("util/eval_metrics.jl")

        
        # STRUCTS
    export GlobalModel, BlackBoxLearner, BlackBoxClassifier, BlackBoxRegressor,ClosedFormConstraint, 
        LinkedLearner, LinkedClassifier, LinkedRegressor, SVM_Classifier, GBM_Classifier, MLP_Classifier,

    # JuMP.Model extensions to GlobalModel
        set_optimizer, optimize!,

        # PROBLEM CONSTRUCTION

        # Explicit constraints
        add_nonlinear_constraint, add_nonlinear_or_compatible, 
        add_linked_constraint,

        # Data driven constraints
        add_variables_from_data!, add_datadriven_constraint, bound_to_data!,

        # CONSTRAINT LEARNING

        # Bound finding
        find_bounds!, find_linear_bounds!,

        # Sampling methods (called on BBLs)
        lh_sample, boundary_sample, last_leaf_sample,
        feasibility_sample, upper_bound_sample,
        uniform_sample_and_eval!, knn_sample, 
        build_knn_tree,
        find_knn, classify_patches,
        opt_sample,opt_sample_helper,

        # Learner methods 
        fit!, predict,

        # Evaluation
        evaluate, evaluate_gradient, update_gradients,
        classify_curvature, eval!, add_data!, make_feasible,

        # Convexity checks
        update_local_convexity, update_vexity, update_leaf_vexity,
        solution, evaluate_accuracy,
        feas_gap, print_feas_gaps,
        nonlinearize!,
        set_param, get_param, 

        # Tree generation and learning
        learn_constraint!, 
        base_regressor, base_classifier, base_rf_regressor, base_rf_classifier,
        fit_regressor_kwargs, fit_classifier_kwargs, 
        regressor_kwargs, classifier_kwargs,

        # Error computation

        # Constraint/variable generation/deletion
        add_feas_constraints!, add_regr_constraints!,
        add_tree_constraints!, clear_tree_constraints!, update_tree_constraints!,
        clear_lower_constraints!, clear_upper_constraints!,
        add_relaxation_variables!, clear_relaxation_variables!,

        # BlackBoxLearner helpers
        show_trees, all_mi_constraints, active_lower_tree, active_upper_tree,
        active_leaves, find_leaves, get_random_trees,

        # Algorithms
        add_infeasibility_cuts!, boundify, surveysolve,
        relax_objective!, tight_objective!,
        globalsolve!, globalsolve_and_time!, descend!, final_sample_repair!,

        # Checks and exceptions
        feasibility, check_accuracy, check_feasibility, 
        check_bounds,  check_sampled, check_if_trained,
        OCTHaGOnException, is_feasible, is_sampled, 

        # Clean-up
        clear_data!, clear_tree_data!,

        # HELPERS
        roc_auc_score,

        # Calculate hash 
        calculate_hash,

        # IAI helpers, 
        pwl_constraint_data, trust_region_data,

        # Functions on JuMP/MathOptInterface objects
        fetch_variable, get_bounds, get_unbounds,
        bounded_aux, 
        linearize_objective!, classify_constraints,
        bound!, restrict_to_set,
        distance_to_set, get_constant, 

        # Data manipulation
        data_to_DataFrame, data_to_Dict,

        # Expr manipulation
        functionify, gradientify, 

        # Other approximators, 
        ridge_regress, u_regress, l_regress, svm, reweight,

        # Functions to import global optimization problems,
        sagemark_to_GlobalModel,
        alphac_to_expr, alphac_to_objexpr, alphac_to_varbound!,

        # Small scripts
        vars_from_expr, vars_from_constraint, flat,
        power, sqr, normalized_data, print_details
end

