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
    function add_relaxation_variables(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    function add_relaxation_variables(gm::GlobalModel, bbls::Array)
Populates relaxvar attributes of all substructs. 
"""
function add_relaxation_variables(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if isnothing(bbl.relax_var)
        bbl.relax_var = @variable(gm.model)
        @constraint(gm.model, bbl.relax_var >= 0)  
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            add_relaxation_variables(gm, ll)
        end
    end
end

function add_relaxation_variables(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        add_relaxation_variables(gm, bbl)
    end
end

add_relaxation_variables(gm::GlobalModel) = add_relaxation_variables(gm, gm.bbls)