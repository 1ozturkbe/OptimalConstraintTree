module OptimalConstraintTree

    using Main.IAI

    include("augment.jl")
    include("black_box_function.jl")
    include("bin_to_leaves.jl")
    include("constraintify.jl")
    include("convexRegress.jl")
    include("fit.jl")
    include("learners.jl")
    include("model_data.jl")
    include("plot.jl")
    include("tools.jl")

           # Structs
    export ModelData, BlackBoxFn,
           # Functions on IAI objects
           gridify, learn_constraints!, learn_from_data!, find_bounds!,
           # Functions on BlackBoxFns
           eval!, sample_and_eval!, plot
           add_fn!, add_linear_ineq!, add_linear_eq!,
           # Functions on JuMP.Models
           add_feas_constraints!, add_regr_constraints!,
           add_linear_constraints!, add_tree_constraints!,
           base_otr, base_otc, update_bounds!, sample, jump_it!,
           sagemark_to_ModelData
end

