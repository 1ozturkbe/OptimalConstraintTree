valid_optimizers = [:Cbc, :CPLEX, :Gurobi, :GLPK, :Mosek, :SCIP]
if !haskey(ENV, "OCTHaGOn_SOLVER")
    loadedModules(m::Module = Main) = filter(x -> eval(x) isa Module && x ≠ Symbol(m), names(m, imported = true))
    pkgs = loadedModules()

    optims = intersect(pkgs, valid_optimizers)

    if isempty(optims)
        throw(ErrorException("OCTHaGOn found no valid mixed-integer optimizers. " *
        "Please choose an optimizer among $(valid_optimizers), and call `Pkg.add(*); using *` " *
        "before calling OCTHaGOn."))
    elseif length(optims) == 1
        ENV["OCTHaGOn_SOLVER"] = String(optims[1])
        @info "OCTHaGOn is using " * ENV["OCTHaGOn_SOLVER"] * " as its mixed-integer optimizer." 
    elseif :CPLEX in optims
        ENV["OCTHaGOn_SOLVER"] = "CPLEX"
        @info "OCTHaGOn found multiple mixed-integer optimizers. Using CPLEX by default. " 
    else
        ENV["OCTHaGOn_SOLVER"] = String(optims[1])
        @info "OCTHaGOn found multiple mixed-integer optimizers. Using " * ENV["OCTHaGOn_SOLVER"] * ", the first available."
    end
end