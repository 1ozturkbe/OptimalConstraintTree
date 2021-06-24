include("../load.jl")

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    uniform_sample_and_eval!(gm)
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)
    for bbl in gm.bbls
        learn_constraint!(bbl)
        add_tree_constraints!(gm, bbl)
    end
    optimize!(gm)    
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while (gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
end 

# Loading BARON examples (that haven't been loaded yet)
for i in ["nlp1.jl", "nlp2.jl", "nlp3.jl"]
    include(OCT.BARON_DIR * i)
end

BARON_PROBLEMS = [minlp, nlp1, nlp2, nlp3]

function optimize_and_time!(m::Union{JuMP.Model, GlobalModel})
    t1 = time()
    if m isa JuMP.Model
        optimize!(m);
    else
        recipe(m)
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
end

bms = [p(false) for p in BARON_PROBLEMS]
optimize_and_time!.(bms)

gms = [p(true) for p in BARON_PROBLEMS]
optimize_and_time!.(gms)

# save_fit.(gms)
# load_fit.(gms)