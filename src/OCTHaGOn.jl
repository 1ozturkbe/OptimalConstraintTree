include("check_solvers.jl")
module OCTHaGOn
    using Combinatorics
    using CPLEX
    using DataFrames
    using DocStringExtensions
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

    const MOI = MathOptInterface
    const VALID_OPTIMIZERS = [:Cbc, :CPLEX, :Gurobi, :GLPK, :Mosek, :SCIP]
    const SOLVER = CPLEX.Optimizer
    const SOLVER_SILENT = with_optimizer(SOLVER, CPX_PARAM_SCRIND = 0) 

    # function silenced_solver(solver::MOI.AbstractOptimizer)
    #     return with_optimizer(solver, CPX_PARAM_SCRIND = 0) 
    # end
    # const SOLVER_SILENT = silenced_solver(SOLVER)

    const PROJECT_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = PROJECT_ROOT * "\\data\\"
    const BARON_DIR = PROJECT_ROOT * "\\data\\baron\\"
    const GAMS_DIR = PROJECT_ROOT * "\\data\\gams\\"  
    const SOL_DIR = PROJECT_ROOT * "\\data\\solutions\\"
    const TREE_DIR = PROJECT_ROOT * "\\data\\trees\\"

    const valid_lowers = ["reg", "lower", "upperlower"]
    const valid_uppers = ["reg", "upper", "upperlower"]
    const valid_singles = ["reg", "upperlower"]
    const valid_pairs =  ["lower" => "upper",
                          "upper" => "lower",
                          "reg" => "upper", # reg with some threshold.second
                          "upper" => "reg"] # reg with some threshold.second

    include("small_scripts.jl")

    include("on_jump.jl")

    include("learners.jl")

    include("exceptions.jl")

    include("settings.jl")
    
    include("black_box_learners.jl")

    include("root_finding.jl")

    include("iai_wrappers.jl")

    include("global_model.jl")
    
    include("data_driven.jl")

    include("sample.jl")

    include("constraintify.jl")

    include("fit.jl")

    include("refine.jl")

    include("optimize.jl")

    include("plot.jl")
        
        # STRUCTS
    export GlobalModel, BlackBoxLearner, BlackBoxClassifier, BlackBoxRegressor, 
        LinkedLearner, LinkedClassifier, LinkedRegressor,

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

        # Evaluation
        evaluate, evaluate_gradient, update_gradients,
        classify_curvature, eval!, add_data!, 

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

        # Constraint/variable generation/deletion
        set_bigM,
        add_feas_constraints!, add_regr_constraints!,
        add_tree_constraints!, clear_tree_constraints!, update_tree_constraints!,
        clear_lower_constraints!, clear_upper_constraints!,
        add_relaxation_variables!, clear_relaxation_variables!,
        detect_linked_constraints!, merge_linked_constraints!,

        # BlackBoxLearner helpers
        show_trees, all_mi_constraints, active_lower_tree, active_upper_tree,
        active_leaves, find_leaves, get_random_trees,

        # Algorithms
        add_infeasibility_cuts!, boundify, surveysolve,
        relax_objective!, tight_objective!,
        globalsolve!, globalsolve_and_time!, descend!,
        compute_hyperplane_bigM_upper, compute_hyperplane_bigM_lower, 
        compute_regression_bigM_lower, compute_regression_bigM_upper,

        # Checks and exceptions
        feasibility, check_accuracy, check_feasibility, 
        check_bounds,  check_sampled, check_if_trained,
        OCTHaGOnException, is_feasible, is_sampled, 

        # Clean-up
        clear_data!, clear_tree_data!,

        # HELPERS

        # IAI helpers, 
        pwl_constraint_data, trust_region_data,

        # Functions on JuMP/MathOptInterface objects
        fetch_variable, get_bounds, get_joint_bounds, 
        get_unbounds, flattened_bounds,
        bounded_aux, 
        linearize_objective!, classify_constraints,
        bound!, restrict_to_set,
        distance_to_set, get_constant,
        count_constraints, count_variables, count_types,

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
        deconstruct, flat, get_datamap, get_var_ranges, get_varmap,
        power, sqr, merge_kwargs, normalized_data, print_details, substitute, vars_from_expr, vars_from_constraint 
    end

