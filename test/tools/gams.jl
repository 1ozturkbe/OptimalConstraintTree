using GAMSFiles
using DataStructures

""" Turns GAMSFiles.GCall into an Expr. """
function eq_to_expr(eq::GAMSFiles.GCall, sets::Dict{String, Any})
    @assert(length(eq.args)==2)
    lhs, rhs, op = eq.args[1], eq.args[2], GAMSFiles.eqops[GAMSFiles.getname(eq)]
    lhsexpr = convert(Expr, lhs)
    rhsexpr = convert(Expr, rhs)
    GAMSFiles.replace_reductions!(lhsexpr, sets)
    GAMSFiles.replace_reductions!(rhsexpr, sets)
    if op in [:(=), :>]
        :($(lhsexpr) - $(rhsexpr))
    elseif op == :<
        :($(rhsexpr) - $(lhsexpr))
    else
        error("unexpected operator ", op)
    end
end

""" Returns whether a GAMSFiles.GCall is an equality. """
function is_equality(eq::GAMSFiles.GCall)
    @assert(length(eq.args)==2)
    GAMSFiles.eqops[GAMSFiles.getname(eq)] == :(=)
end

""" Finds and returns all varkeys in equation. """
function find_vars_in_eq(eq::GAMSFiles.GCall, vardict::Dict{Symbol, Any})
    vars = Symbol[]
    lhs, rhs = eq.args[1], eq.args[2]
    for (var, vinfo) in vardict
        if GAMSFiles.hasvar(lhs, string(var)) || GAMSFiles.hasvar(rhs, string(var))
            push!(vars, var)
        end
    end
    return vars
end

""" Takes gams and turns them into JuMP.Variables"""
function generate_variables!(model::JuMP.Model, gams::Dict{String, Any})
    gamsvars = GAMSFiles.allvars(gams)
    vardict = Dict{Symbol, Any}()
    constdict = Dict{Symbol, Any}()
    for (var, vinfo) in gamsvars
        if vinfo isa Union{Array, Real}
            model[Symbol(var)] = vinfo
            constdict[Symbol(var)] = vinfo
        elseif vinfo isa Base.RefValue
            model[Symbol(var)] = vinfo.x
            constdict[Symbol(var)] = vinfo.x
        else
            axs = vinfo.axs
            if axs == ()
                nv = @variable(model, base_name = var)
                model[Symbol(var)] = nv
                vardict[Symbol(var)] = nv
            else
                nv = JuMP.Containers.DenseAxisArray{JuMP.VariableRef}(undef, axs...)
                for idx in eachindex(nv)
                    nv[idx] = @variable(model)
                    set_name(nv[idx], "$(var)[$(join(Tuple(idx),","))]")
                end
                model[Symbol(var)] = nv
                vardict[Symbol(var)] = nv
            end
            if vinfo.typ != "free"
                if vinfo.typ == "positive"
                    JuMP.set_lower_bound.(model[Symbol(var)], 0)
                elseif vinfo.typ == "negative"
                    JuMP.set_upper_bound.(model[Symbol(var)], 0)
                elseif vinfo.typ == "binary"
                    JuMP.set_binary.(model[Symbol(var)])
                    JuMP.set_lower_bound.(model[Symbol(var)], 0)
                    JuMP.set_upper_bound.(model[Symbol(var)], 1)
                elseif vinfo.typ == "integer"
                    JuMP.set_integer.(model[Symbol(var)])
                else
                    throw(OCTHaGOnException("Type $(vinfo.typ) unknown for variable $(var)."))
                end
            end
            for (prop, val) in vinfo.assignments
                if hasproperty(prop, :indices)
                    inds = map(x->x.val, prop.indices)
                    nv = model[Symbol(var)][inds...]
                    c = val.val
                    if prop.name =="l"
                        JuMP.set_start_value(nv, c)
                    elseif prop.name == "fx"
                        JuMP.set_lower_bound(nv, c)
                        JuMP.set_upper_bound(nv, c)
                    elseif prop.name == "lo"
                        JuMP.set_lower_bound(nv, c)
                    elseif prop.name == "up"
                        JuMP.set_upper_bound(nv, c)
                    end
                else
                    nv = model[Symbol(var)]
                    c = val.val
                    if prop.text == "l"
                        JuMP.set_start_value(nv, c)
                    elseif prop.text == "fx"
                        JuMP.set_lower_bound(nv, c)
                        JuMP.set_upper_bound(nv, c)
                    elseif prop.text == "lo"
                        JuMP.set_lower_bound(nv, c)
                    elseif prop.text == "up"
                        JuMP.set_upper_bound(nv, c)
                    end
                end
            end
        end
    end
    return vardict, constdict
end

""" 
    GAMS_to_GlobalModel(GAMS_DIR::String, filename::String)

Converts a GAMS optimization model to a GlobalModel.
GAMS_DIR is the directory to look in, while the filename
is the name of the .gms file. 
"""
function GAMS_to_GlobalModel(GAMS_DIR::String, filename::String; alg_list=["OCT"], regression=false, relax_coeff=0)
    
    #relax_coeff = 1
    use_relax_var = relax_coeff != 0

    model = JuMP.Model()
    
    # Parsing GAMS Files
    lexed = GAMSFiles.lex(GAMS_DIR * filename)
    gams = GAMSFiles.parsegams(GAMS_DIR * filename)
    GAMSFiles.parseconsts!(gams)
    
    vars = GAMSFiles.getvars(gams["variables"])
    sets = Dict{String, Any}()
    if haskey(gams, "sets")
        sets = gams["sets"]
    end
    preexprs, bodyexprs = Expr[], Expr[]
    if haskey(gams, "parameters") && haskey(gams, "assignments")
        GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
    end
    
    # Getting variables
    vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables
    obj_symb = :objvar

    
    #relax_var = use_relax_var ? @variable(model, r_rel>=0) : nothing
    relax_var = @variable(model, r_rel>=0)
    relax_term = use_relax_var ? relax_coeff*relax_var : 0;
    print("Relax term is: $(relax_term)")

    unrelaxed_obj = nothing 
    objective_sense = "Min" 

    # Getting objective
    if "minimizing" in keys(gams)
        if gams["minimizing"] isa String
            unrelaxed_obj = JuMP.variable_by_name(model, gams["minimizing"])
            @objective(model, Min, unrelaxed_obj+relax_term)
            obj_symb = Symbol(gams["minimizing"])
        else
            unrelaxed_obj = sum([JuMP.variable_by_name(i) for i in gams["minimizing"]])
            @objective(model, Min, unrelaxed_obj+relax_term)
        end
        objective_sense = "Min"
    elseif "maximizing" in keys(gams)
        @warn "$(filename) is a maximization. Make sure objvar is on LHS of constraints, with a positive coefficient and and equality."
        if gams["maximizing"] isa String
            unrelaxed_obj = JuMP.variable_by_name(model, gams["maximizing"])
            @objective(model, Max, unrelaxed_obj-relax_term)
            obj_symb = Symbol(gams["maximizing"])
        else
            unrelaxed_obj = sum([JuMP.variable_by_name(i) for i in gams["maximizing"]])
            @objective(model, Max, unrelaxed_obj-relax_term)
        end
        objective_sense = "Min"
    end

    # Creating GlobalModel
    gm = GlobalModel(model = model, name = replace(filename, ".gms" => ""))
    gm.relax_var = relax_var 
    gm.relax_coeff = relax_coeff
    gm.objective = unrelaxed_obj
    gm.objective_sense = objective_sense

    # Creating GlobalModel
    equations = [] # For debugging purposes...
    for (key, eq) in gams["equations"]
        push!(equations, key => eq)
    end
    for (key, eq) in equations
        if key isa GAMSFiles.GText
            constr_expr = eq_to_expr(eq, sets)
            # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(constkey => model[constkey] for constkey in constkeys)
            for (constkey, constval) in const_pairs
                constr_expr = OCTHaGOn.substitute(constr_expr, :($constkey) => constval)
            end
            # Designate free variables
            varkeys = find_vars_in_eq(eq, vardict)
            if !(obj_symb in varkeys)
                vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
                input = Symbol.(varkeys)
                constr_fn = :(($(input...),) -> $(constr_expr))
                if length(input) == 1
                    constr_fn = :($(input...) -> $(constr_expr))
                end
                add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                                        equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key), alg_list = alg_list, regression = regression)
            else
                constr_expr = OCTHaGOn.substitute(constr_expr, :($(obj_symb)) => 0)
                # ASSUMPTION: objvar has positive coefficient, and is on the greater size. 
                op = GAMSFiles.eqops[GAMSFiles.getname(eq)]
                # if !(op in [:<, :>])
                    # throw(OCTHaGOnException("Please make sure GAMS model has objvar on the greater than size of inequalities, " *
                    #                     " with a leading coefficient of 1."))
                # end
                varkeys = filter!(x -> x != obj_symb, varkeys)
                vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
                input = Symbol.(varkeys)
                constr_fn = :(($(input...),) -> -$(constr_expr))
                if length(input) == 1
                    constr_fn = :($(input...) -> -$(constr_expr))
                end
                add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                    dependent_var = vardict[obj_symb], equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key), alg_list = alg_list, regression = regression)
            end
        elseif key isa GAMSFiles.GArray
            axs = GAMSFiles.getaxes(key.indices, sets)
            idxs = collect(Base.product([collect(ax) for ax in axs]...))
            names = [gm.name * "_" * key.name * string(idx) for idx in idxs]
            constr_expr = eq_to_expr(eq, sets)
                    # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(Symbol(constkey) => model[constkey] for constkey in constkeys)
            for (constkey, constval) in const_pairs
                constr_expr = OCTHaGOn.substitute(constr_expr, :($constkey) => constval)
            end
            # Designate free variables
            varkeys = find_vars_in_eq(eq, vardict)
            vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
            input = Symbol.(varkeys)
            constr_fn = :(($(input...),) -> $(constr_expr))
            if length(input) == 1
                constr_fn = :($(input...) -> $(constr_expr))
            end
            constr_fns = []
            for idx in idxs
                new_fn = copy(constr_fn)
                for ax_number in 1:length(axs)
                    new_fn = OCTHaGOn.substitute(new_fn, Symbol(key.indices[ax_number].text) => idx[ax_number]);
                end
                push!(constr_fns, new_fn)
            end
            for i = 1:length(idxs)
                add_nonlinear_or_compatible(gm, constr_fns[i], vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                                         equality = is_equality(eq),
                                         name = names[i], alg_list = alg_list, regression = regression)
            end
        end
    end
    return gm
end

function getIndex(s, i)
  
    # If input is invalid.
    if s[i:i] != "("
        return -1
    end
  
    # Create a deque to use it as a stack.
    d = Deque{String}()
  
    # Traverse through all elements
    # starting from i.
    for k in i:length(s)
  
        # Pop a starting bracket
        # for every closing bracket
        if s[k:k] == ")"
            popfirst!(d)
        # Push all starting brackets
        elseif s[k:k] == "("
            push!(d, s[i:i])
        end
        
        # If deque becomes empty
        if isempty(d)
            return k
        end
    end
    return -1
end

function replace_tokens(s)
    # try
        if length(s) == 0
            return s
        else
            cont = true

            while cont

                cont = false

                for token in ["power", "sqr"]

                    idxes = findfirst(token, s)

                    if !isnothing(idxes)
                        par_idx = idxes[end]+1
                        close_par_idx = getIndex(s, par_idx)

                        # sqr
                        if token == "sqr"
                            inner_expr = replace_tokens(s[par_idx+1:close_par_idx-1])
                            s = s[1:idxes[1]-1]*"("*inner_expr*")^2"*s[close_par_idx+1:end]
                        elseif token == "power"
                            inner_expr = replace_tokens(s[par_idx+1:close_par_idx-1])
                            comma_idx = findfirst(",", inner_expr)[1]
                            s = s[1:idxes[1]-1]*"("*inner_expr[1:comma_idx-1]*")^"*inner_expr[comma_idx+1:end]*s[close_par_idx+1:end]
                        end

                        # Continue to check for more matches
                        cont = true
                    end
                end
            end
        end
        return s
    # catch
    #     println("Error parsing expression $(s)")
    #     return s
    # end
end

function GAMS_to_baron_model(GAMS_DIR::String, filename::String)
    global model = JuMP.Model(with_optimizer(BARON.Optimizer, PrLevel=1, MaxTime=1500))
    # Parsing GAMS Files
    lexed = GAMSFiles.lex(GAMS_DIR * filename)
    gams = GAMSFiles.parsegams(GAMS_DIR * filename)
    GAMSFiles.parseconsts!(gams)
    #println(alg_list)
    vars = GAMSFiles.getvars(gams["variables"])

    sets = Dict{String, Any}()
    if haskey(gams, "sets")
        sets = gams["sets"]
    end
    preexprs, bodyexprs = Expr[], Expr[]
    if haskey(gams, "parameters") && haskey(gams, "assignments")
        GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
    end

    # Getting variables
    global vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables

    # Getting objective
    if gams["minimizing"] isa String
        @objective(model, Min, JuMP.variable_by_name(model, gams["minimizing"]))
    else
        @objective(model, Min, sum([JuMP.variable_by_name(i) for i in gams["minimizing"]]))
    end

    # Creating GlobalModel
    #gm = OCTHaGOn.GlobalModel(model = model, name = replace(filename, ".gms" => ""))
    equations = [] # For debugging purposes...
    for (key, eq) in gams["equations"]
        push!(equations, key => eq)
    end

    # Register the variables as they appear in the constraints
    for (key, var) in vardict
        eval(Meta.parse("global $(key) = vardict[:$(key)]"))
        #eval(Meta.parse("$(key) = 0"))
    end
    #x31 = 1

    fn = nothing
    varkeys = nothing
    constr_fn = nothing
    constr_expr = nothing
    vars = nothing
    for (key, eq) in equations#[1:end-1]
        if key isa GAMSFiles.GText
            constr_expr = eq_to_expr(eq, sets)
            #println(constr_expr)
            # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(constkey => model[constkey] for constkey in constkeys)
            for (constkey, constval) in const_pairs
                constr_expr = OCTHaGOn.substitute(constr_expr, :($constkey) => constval)
            end
            # Designate free variables
            varkeys = find_vars_in_eq(eq, vardict)
            if !(Symbol(gams["minimizing"]) in varkeys)
                vars = Array{VariableRef}(OCTHaGOn.flat([vardict[varkey] for varkey in varkeys]))
                input = Symbol.(varkeys)
                constr_fn = :(($(input...),) -> $(constr_expr))
                if length(input) == 1
                    constr_fn = :($(input...) -> $(constr_expr))
                end
                
                s_constr = string(constr_expr)
                s_constr = replace_tokens(s_constr)
                #s_constr = replace(s_constr, r"power\(\s*([^,\s)]+)\s*,\s*([0-9]+)\s*\)" => s"\g<1>^\g<2>")
                #s_constr = replace(s_constr, r"sqr\(\s*([^\s)]+)\s*\)" => s"\g<1>^2")
                
                if is_equality(eq)
                    #@constraint(m, Base.invokelatest(eval(constr_fn), vars...)==0)
                    eval(Meta.parse("@NLconstraint(model, ("*s_constr*") == 0)"))
                else
                    #@constraint(m, Base.invokelatest(eval(constr_fn), vars...)>=0)
                    eval(Meta.parse("@NLconstraint(model, ("*s_constr*") >= 0)"))
                end
                #OCTHaGOn.add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                #                        equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key), alg_list = alg_list, regression = regression)
            else
                constr_expr = OCTHaGOn.substitute(constr_expr, :($(Symbol(gams["minimizing"]))) => 0)
                # ASSUMPTION: objvar has positive coefficient, and is on the greater size. 
                op = GAMSFiles.eqops[GAMSFiles.getname(eq)]
                # if !(op in [:<, :>])
                #     throw(OCTHaGOnException("Please make sure GAMS model has objvar on the greater than size of inequalities, " *
                #                         " with a leading coefficient of 1."))
                # end
                varkeys = filter!(x -> x != Symbol(gams["minimizing"]), varkeys)
                vars = Array{VariableRef}(OCTHaGOn.flat([vardict[varkey] for varkey in varkeys]))
                input = Symbol.(varkeys)
                constr_fn = :(($(input...),) -> -$(constr_expr))
                if length(input) == 1
                    constr_fn = :($(input...) -> -$(constr_expr))
                end
                obj_var = vardict[Symbol(gams["minimizing"])]
                s_constr = string(constr_expr)
                s_constr = replace_tokens(s_constr)

                #s_constr = replace(s_constr, r"power\(\s*([^,\s)]+)\s*,\s*([0-9]+)\s*\)" => s"\g<1>^\g<2>")
                #s_constr = replace(s_constr, r"sqr\(\s*([^\s)]+)\s*\)" => s"\g<1>^2")
                #println(s_constr)
                eval(Meta.parse("@NLconstraint(model, -("*string(s_constr)*") == $(obj_var))"))
                #@NLconstraint(m, Base.invokelatest(eval(constr_fn), vars...) == vardict[Symbol(gams["minimizing"])])
    #             OCTHaGOn.add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
    #                 dependent_var = vardict[Symbol(gams["minimizing"])], equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key), alg_list = alg_list, regression = regression)
            end
            fn = constr_fn
        elseif key isa GAMSFiles.GArray
            axs = GAMSFiles.getaxes(key.indices, sets)
            idxs = collect(Base.product([collect(ax) for ax in axs]...))
            #names = [gm.name * "_" * key.name * string(idx) for idx in idxs]
            constr_expr = OCTHaGOn.eq_to_expr(eq, sets)
                    # Substitute constant variables
            constkeys = OCTHaGOn.find_vars_in_eq(eq, constdict)
            const_pairs = Dict(Symbol(constkey) => model[constkey] for constkey in constkeys)
            for (constkey, constval) in const_pairs
                constr_expr = OCTHaGOn.substitute(constr_expr, :($constkey) => constval)
            end
            # Designate free variables
            varkeys = OCTHaGOn.find_vars_in_eq(eq, vardict)
            vars = Array{VariableRef}(OCTHaGOn.flat([vardict[varkey] for varkey in varkeys]))
            input = Symbol.(varkeys)
            constr_fn = :(($(input...),) -> $(constr_expr))
            if length(input) == 1
                constr_fn = :($(input...) -> $(constr_expr))
            end
            constr_fns = []
            for idx in idxs
                new_fn = copy(constr_fn)
                for ax_number in 1:length(axs)
                    new_fn = OCTHaGOn.substitute(new_fn, Symbol(key.indices[ax_number].text) => idx[ax_number]);
                end
                push!(constr_fns, new_fn)
            end
    #         for i = 1:length(idxs)
    #             OCTHaGOn.add_nonlinear_or_compatible(gm, constr_fns[i], vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
    #                                      equality = is_equality(eq),
    #                                      name = names[i], alg_list = alg_list, regression = regression)
    #         end
        end
    end
    return model
end

# """ Solver wrapper for GAMS benchmarking. """
# function nonlinear_solve(gm::GlobalModel; solver = IPOPT_SILENT)
#     nonlinearize!(gm)
#     set_optimizer(gm, solver)
#     optimize!(gm)
# end