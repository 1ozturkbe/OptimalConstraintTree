var documenterSearchIndex = {"docs":
[{"location":"api/iai_wrappers.html#IAI-wrappers","page":"IAI wrappers","title":"IAI wrappers","text":"","category":"section"},{"location":"api/iai_wrappers.html","page":"IAI wrappers","title":"IAI wrappers","text":"Work in progress...","category":"page"},{"location":"api/iai_wrappers.html","page":"IAI wrappers","title":"IAI wrappers","text":"find_leaves\r\npwl_constraint_data\r\ntrust_region_data\r\ncheck_if_trained","category":"page"},{"location":"api/iai_wrappers.html#Main.OCTHaGOn.find_leaves","page":"IAI wrappers","title":"Main.OCTHaGOn.find_leaves","text":"find_leaves(lnr::OptimalTreeLearner)\n\nFinds all leaves of OptimalTreeLearner.\n\n\n\n\n\n","category":"function"},{"location":"api/iai_wrappers.html#Main.OCTHaGOn.pwl_constraint_data","page":"IAI wrappers","title":"Main.OCTHaGOn.pwl_constraint_data","text":"pwl_constraint_data(lnr::IAI.OptimalTreeLearner, vks)\n\nCreates PWL dataset from a OptimalTreeLearner Arguments:     lnr: OptimalTreeLearner     vks: headers of DataFrame X, i.e. varkeys Returns:     Dict[leaf_number] containing [B0, B]\n\n\n\n\n\n","category":"function"},{"location":"api/iai_wrappers.html#Main.OCTHaGOn.trust_region_data","page":"IAI wrappers","title":"Main.OCTHaGOn.trust_region_data","text":"trust_region_data(lnr:: IAI.OptimalTreeLearner, vks)\n\nCreates trust region from a OptimalTreeLearner Arguments:     lnr: OptimalTreeLearner     vks: headers of DataFrame X, i.e. varkeys Returns:     Dict[leaf_number] containing [B0, B]\n\n\n\n\n\n","category":"function"},{"location":"api/iai_wrappers.html#Main.OCTHaGOn.check_if_trained","page":"IAI wrappers","title":"Main.OCTHaGOn.check_if_trained","text":"Checks if a learner is trained. \n\n\n\n\n\n","category":"function"},{"location":"api/datastructures.html#Data-Structures","page":"Data Structures","title":"Data Structures","text":"","category":"section"},{"location":"api/datastructures.html","page":"Data Structures","title":"Data Structures","text":"Work in progress...","category":"page"},{"location":"api/datastructures.html","page":"Data Structures","title":"Data Structures","text":"GlobalModel\r\nBlackBoxLearner\r\nBlackBoxClassifier\r\nBlackBoxRegressor\r\nLinkedLearner\r\nLinkedClassifier\r\nLinkedRegressor","category":"page"},{"location":"api/datastructures.html#Main.OCTHaGOn.GlobalModel","page":"Data Structures","title":"Main.OCTHaGOn.GlobalModel","text":"Contains all required info to be able to generate a global optimization problem. NOTE: proper construction is to use addnonlinearconstraint to add bbls. model must be a mixed integer convex model. nonlinear_model can contain JuMP.NonlinearConstraints.\n\n\n\n\n\n","category":"type"},{"location":"api/datastructures.html#Main.OCTHaGOn.BlackBoxLearner","page":"Data Structures","title":"Main.OCTHaGOn.BlackBoxLearner","text":"BBL type is for function definitions! \n\n\n\n\n\n","category":"type"},{"location":"api/datastructures.html#Main.OCTHaGOn.BlackBoxClassifier","page":"Data Structures","title":"Main.OCTHaGOn.BlackBoxClassifier","text":"@with_kw mutable struct BlackBoxClassifier\n\nAllows for approximation of constraints using OCTs. To be added to GlobalModel.bbls using functions:     addnonlinearconstraints     addnonlinearor_compatible\n\nMandatory arguments are:     vars::Array{JuMP.VariableRef,1}\n\nOther arguments may be necessary for proper functioning:     For data-driven constraints, need:         X::DataFrame         Y:: Array     For constraint functions, need :         constraint::Union{JuMP.ConstraintRef, Expr}\n\nOptional arguments:     exprvars::Union{Array, Nothing}         JuMP variables as function arguments (i.e. vars rolled up into vector forms).         vars ⋐ flat(exprvars)     name::String     equality::Bool         Specifies whether function should be satisfied to an equality\n\n\n\n\n\n","category":"type"},{"location":"api/datastructures.html#Main.OCTHaGOn.BlackBoxRegressor","page":"Data Structures","title":"Main.OCTHaGOn.BlackBoxRegressor","text":"@with_kw mutable struct BlackBoxRegressor\n\nAllows for approximation of constraints using OCTs. To be added to GlobalModel.bbls using functions:     addnonlinearconstraints     addnonlinearor_compatible\n\nMandatory arguments are:\n    vars::Array{JuMP.VariableRef,1}\n    dependent_var::JuMP.VariableRef\n\nOther arguments may be necessary for proper functioning:     For data-driven constraints, need:         X::DataFrame         Y:: Array     For constraint functions, need :         constraint::Union{JuMP.ConstraintRef, Expr}\n\nOptional arguments:     exprvars::Union{Array, Nothing}         JuMP variables as function arguments (i.e. vars rolled up into vector forms).         vars ⋐ flat(exprvars)     name::String     equality::Bool         Specifies whether function should be satisfied to an equality\n\n\n\n\n\n","category":"type"},{"location":"api/datastructures.html#Main.OCTHaGOn.LinkedLearner","page":"Data Structures","title":"Main.OCTHaGOn.LinkedLearner","text":"Superclass of LinkedClassifier and LinkedRegressor.\n\n\n\n\n\n","category":"type"},{"location":"api/datastructures.html#Main.OCTHaGOn.LinkedClassifier","page":"Data Structures","title":"Main.OCTHaGOn.LinkedClassifier","text":"Contains data for a constraint that is repeated. \n\n\n\n\n\n","category":"type"},{"location":"api/datastructures.html#Main.OCTHaGOn.LinkedRegressor","page":"Data Structures","title":"Main.OCTHaGOn.LinkedRegressor","text":"Contains data for a constraint that is repeated. \n\n\n\n\n\n","category":"type"},{"location":"api/helpers.html#Helper-functions","page":"Helper functions","title":"Helper functions","text":"","category":"section"},{"location":"api/helpers.html","page":"Helper functions","title":"Helper functions","text":"Work in progress...","category":"page"},{"location":"api/sampling.html#Sampling-methods","page":"Sampling methods","title":"Sampling methods","text":"","category":"section"},{"location":"api/sampling.html","page":"Sampling methods","title":"Sampling methods","text":"Work in progress...","category":"page"},{"location":"api/sampling.html","page":"Sampling methods","title":"Sampling methods","text":"uniform_sample_and_eval!\r\nlh_sample\r\nboundary_sample\r\nknn_sample","category":"page"},{"location":"api/sampling.html#Main.OCTHaGOn.uniform_sample_and_eval!","page":"Sampling methods","title":"Main.OCTHaGOn.uniform_sample_and_eval!","text":"uniform_sample_and_eval!(bbl::Union{BlackBoxLearner, GlobalModel, Array{BlackBoxLearner}};\n                          boundary_fraction::Float64 = 0.5,\n                          lh_iterations::Int64 = 0)\n\nUniform samples and evaluates a BlackBoxLearner. Furthermore, sets the big-M value.  Keyword arguments:     boundaryfraction: maximum ratio of boundary samples     lhiterations: number of GA populations for LHC sampling (0 is a random LH.)\n\n\n\n\n\n","category":"function"},{"location":"api/sampling.html#Main.OCTHaGOn.lh_sample","page":"Sampling methods","title":"Main.OCTHaGOn.lh_sample","text":"lh_sample(vars::Array{JuMP.VariableRef, 1}; lh_iterations::Int64 = 0,\n               n_samples::Int64 = 1000)\nlh_sample(bbl::BlackBoxLearner; lh_iterations::Int64 = 0,\n               n_samples::Int64 = 1000)\n\nUniformly Latin Hypercube samples the variables of GlobalModel, as long as all lbs and ubs are defined.\n\n\n\n\n\n","category":"function"},{"location":"api/sampling.html#Main.OCTHaGOn.boundary_sample","page":"Sampling methods","title":"Main.OCTHaGOn.boundary_sample","text":"boundary_sample(bbl::BlackBoxLearner; fraction::Float64 = 0.5)\nboundary_sample(vars::Array{JuMP.VariableRef, 1}; n_samples = 100, fraction::Float64 = 0.5,\n                     warn_string::String = \"\")\n\nSmartly samples the constraint along the variable boundaries.     NOTE: Because we are sampling symmetrically for lower and upper bounds,     the choose coefficient has to be less than ceil(half of number of dims).\n\n\n\n\n\n","category":"function"},{"location":"api/sampling.html#Main.OCTHaGOn.knn_sample","page":"Sampling methods","title":"Main.OCTHaGOn.knn_sample","text":"knn_sample(bbl::BlackBoxClassifier; k::Int64 = 10, sample_density = 1e-5, sample_idxs = nothing)\n\nDoes KNN and secant method based sampling once there is at least one feasible     sample to a BlackBoxLearner.\n\n\n\n\n\n","category":"function"},{"location":"basic.html#Basic-usage","page":"Basic usage","title":"Basic usage","text":"","category":"section"},{"location":"basic.html","page":"Basic usage","title":"Basic usage","text":"Work in progress...","category":"page"},{"location":"installation.html#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"installation.html","page":"Installation","title":"Installation","text":"As an overview of the installation steps, the current version of OCTHaGON works with Julia 1.5, with Interpretable AI as its back-end for constraint learning, and CPLEX as its default solver. Please follow the instructions below to get OCTHaGOn working on your machine. ","category":"page"},{"location":"installation.html#Installing-required-software","page":"Installation","title":"Installing required software","text":"","category":"section"},{"location":"installation.html#Julia","page":"Installation","title":"Julia","text":"","category":"section"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Please find instructions for installing Julia on various platforms here. OCTHaGOn is compatible with Julia 1.5, but is frequently tested on Julia 1.5.4, making that the most robust version. However, if you have an existing Julia v1.5.4, we highly recommend you install a clean v1.5.x before proceeding. ","category":"page"},{"location":"installation.html#Interpretable-AI","page":"Installation","title":"Interpretable AI","text":"","category":"section"},{"location":"installation.html","page":"Installation","title":"Installation","text":"OCTHaGOn requires an installation of Interpretable AI (IAI) for its various machine learning tools. Different builds of IAI are found here, corresponding to the version of Julia used. IAI requires a pre-built system image of Julia to replace the existing image (sys.so in Linux and sys.dll in Windows machines), thus the need for a clean install of Julia v1.5.x. For your chosen v1.5, please replace the system image with the one you downloaded. Then request and deploy an IAI license (free for academics) by following the instructions here. ","category":"page"},{"location":"installation.html#CPLEX","page":"Installation","title":"CPLEX","text":"","category":"section"},{"location":"installation.html","page":"Installation","title":"Installation","text":"CPLEX is a mixed-integer optimizer that can be found here. It is free to solve optimization problems with up to 1000 variables and constraints, available via signing up, and also available via a free academic license for larger problems. ","category":"page"},{"location":"installation.html#Quickest-build","page":"Installation","title":"Quickest build","text":"","category":"section"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Once the above steps are complete, we recommend using the following set of commands as the path of least resistance to getting started. ","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Navigate to where you would like to put OCTHaGOn, and call the following commands to instantiate and check all of the dependencies. ","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"git clone https://github.com/1ozturkbe/OCTHaGOn.jl.git\r\ncd OCTHaGOn.jl\r\njulia --project=.\r\nusing Pkg\r\nPkg.instantiate()","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Call the following to precompile all packages and load OCTHaGOn to your environment:","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"include(\"src/OCTHaGOn.jl\")\r\nusing .OCTHaGOn","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Alternatively, you can test your installation of OCTHaGOn by doing the following in a new Julia terminal:","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"using Pkg\r\nPkg.activate(\"test\")\r\nPkg.instantiate()\r\ninclude(\"test/load.jl\")\r\ninclude(\"test/src.jl\")","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Please see Basic usage for an simple application of OCTHaGOn to a MINLP!","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"(Image: codecov)","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"OCTHaGOn (Optimal Classification Trees with Hyperplanes for Global Optimization) is a Julia package that allows for the solution of global optimization problems using mixed-integer (MI) linear and convex approximations. It is an implementation of the methods detailed in Chapter 2 of this thesis and submitted to Operations Research. OCTHaGOn is licensed under the MIT License. ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"OCTHaGOn leans on the JuMP.jl  modeling language in its backend, and it develops MI approximations using  Interpretable AI, with a free academic license. The problems can then be solved by JuMP-compatible solvers, depending on  the type of approximation. OCT's default solver in tests is CPLEX,  which is free with an academic license as well. ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"This documentation is a work in progress.  If you have any burning questions or applications, or are having problems with OCTHaGOn, please create an issue! ","category":"page"},{"location":"api/constraints.html#Nonlinear-constraints","page":"Nonlinear constraints","title":"Nonlinear constraints","text":"","category":"section"},{"location":"api/constraints.html","page":"Nonlinear constraints","title":"Nonlinear constraints","text":"Work in progress...","category":"page"},{"location":"api/constraints.html","page":"Nonlinear constraints","title":"Nonlinear constraints","text":"add_nonlinear_constraint\r\nadd_nonlinear_or_compatible\r\nadd_linked_constraint","category":"page"},{"location":"api/constraints.html#Main.OCTHaGOn.add_nonlinear_constraint","page":"Nonlinear constraints","title":"Main.OCTHaGOn.add_nonlinear_constraint","text":"add_nonlinear_constraint(gm::GlobalModel,\n                 constraint::Union{JuMP.ConstraintRef, Expr};\n                 vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,\n                 expr_vars::Union{Nothing, Array} = nothing,\n                 dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,\n                 name::String = gm.name * \" \" * string(length(gm.bbls) + 1),\n                 equality::Bool = false)\n\nAdds a new nonlinear constraint to Global Model. Standard method for adding bbls.\n\n\n\n\n\n","category":"function"},{"location":"api/constraints.html#Main.OCTHaGOn.add_nonlinear_or_compatible","page":"Nonlinear constraints","title":"Main.OCTHaGOn.add_nonlinear_or_compatible","text":"add_nonlinear_or_compatible(gm::GlobalModel,\n                     constraint::Union{JuMP.ConstraintRef, Expr};\n                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,\n                     expr_vars::Union{Nothing, Array} = nothing,\n                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,\n                     name::String = gm.name * \"_\" * string(length(gm.bbls) + 1),\n                     equality::Bool = false)\n\nExtents addnonlinearconstraint to recognize JuMP compatible constraints and add them as normal JuMP constraints\n\n\n\n\n\n","category":"function"},{"location":"api/constraints.html#Main.OCTHaGOn.add_linked_constraint","page":"Nonlinear constraints","title":"Main.OCTHaGOn.add_linked_constraint","text":"add_linked_constraint(gm::GlobalModel, bbc::BlackBoxClassifier, linked_vars::Array{JuMP.Variable})\nadd_linked_constraint(gm::GlobalModel, bbr::BlackBoxRegressor, linked_vars::Array{JuMP.Variable}, linked_dependent::JuMP.Variable)\n\nAdds a linked constraint of the same structure as the BBC/BBR.  When a nonlinear constraint is repeated more than once, this function allows the underlying approximator to be replicated without retraining trees for each constraint.   Note that the bounds used for sampling are for the original variables of the BBC/BBR, so be careful!\n\n\n\n\n\n","category":"function"},{"location":"api/constraints.html#Data-driven-constraints","page":"Nonlinear constraints","title":"Data driven constraints","text":"","category":"section"},{"location":"api/constraints.html","page":"Nonlinear constraints","title":"Nonlinear constraints","text":"add_variables_from_data!\r\nadd_datadriven_constraint\r\nbound_to_data!","category":"page"},{"location":"api/constraints.html#Main.OCTHaGOn.add_variables_from_data!","page":"Nonlinear constraints","title":"Main.OCTHaGOn.add_variables_from_data!","text":"add_variables_from_data!(gm::Union{JuMP.Model, GlobalModel},\n                        X::DataFrame)\n\nAdds/finds variables depending on the columns of X. \n\n\n\n\n\n","category":"function"},{"location":"api/constraints.html#Main.OCTHaGOn.add_datadriven_constraint","page":"Nonlinear constraints","title":"Main.OCTHaGOn.add_datadriven_constraint","text":"add_datadriven_constraint(gm::GlobalModel,\n                 X::DataFrame, Y::Array;\n                 constraint::Union{Nothing, JuMP.ConstraintRef, Expr} = nothing, \n                 vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,\n                 dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,\n                 name::String = \"bbl\" * string(length(gm.bbls) + 1),\n                 equality::Bool = false)\n\nAdds a data-driven constraint to GlobalModel. Data driven BBLs do not allow for resampling. \n\n\n\n\n\n","category":"function"},{"location":"api/constraints.html#Main.OCTHaGOn.bound_to_data!","page":"Nonlinear constraints","title":"Main.OCTHaGOn.bound_to_data!","text":"bound_to_data!(gm::Union{JuMP.Model, GlobalModel},\n              X::DataFrame)\n\nConstrains the domain of relevant variables to the box interval defined by X.\n\n\n\n\n\n","category":"function"}]
}
