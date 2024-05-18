
function mse(y_pred,y_true; weights = nothing, squared = true, dims=1)
    #_validate_distance_input(y_true, y_pred, weights)
    result = (y_true .- y_pred) .^ 2
    result = !squared ? sqrt.(result) : result
    return weights == nothing ? mean(result) : mean(result .* weights)
end

function mas(y_pred,y_true; weights = nothing, squared = false, dims=1)
    return mean(abs.(y_true .- y_pred))
end

loss = mse

# Define dense layer:
# Define the dense layer
struct Dense; 
    w; 
    b; 
    f; 
end
Dense(i::Int, o::Int, f = relu) = Dense(param(o, i), param0(o), f); # constructor
(d::Dense)(x) = d.f.(d.w * x .+ d.b); # define method for dense layer


# Define the chain layer
struct MLP; 
    layers::Array{Dense};
    loss::Function;
end
(c::MLP)(x) = (for l in c.layers; x = l(x); end; x); # define method for feed-forward
(c::MLP)(x, y) = c.loss(c(x), y); # define method for mse loss function


@with_kw mutable struct MLP_Regressor <: AbstractRegressor
    # Arguments
    solver = CPLEX_SILENT
    equality::Bool = false
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing
    scaler = nothing 
    train_data = nothing

    # Model variables
    mlp::Union{MLP, Nothing} = nothing
end

@with_kw mutable struct MLP_Classifier <: AbstractClassifier
    # Arguments
    solver = CPLEX_SILENT
    equality::Bool = false
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing

    # Model variables
    mlp::Union{MLP, Nothing} = nothing
    
end


"""
Used to fit an MLP regressor
"""
function fit!(lnr::MLP_Regressor, X::DataFrame, Y::Array; equality=false)

    lnr.equality = equality

    # Convert dataframe to matrix 
    X = Matrix(X)
    lnr.train_data = X
    # lnr.scaler = StatsBase.fit(ZScoreTransform, X, dims=1)
    # X = StatsBase.transform(lnr.scaler, X)

    # Initialize model 
    net = MLP(
        [Dense(size(X, 2), 100), Dense(100, 1, identity)],
        mas
    );
    lnr.mlp = net
    
    # Initialize training set 
    dtrn = minibatch(Float32.(X)', Float32.(Y), 16)

    # Train for 100 epochs
    adam!(lnr.mlp, repeat(dtrn, 500));

end


"""
Used to fit an MLP classifier
"""
function fit!(lnr::MLP_Classifier, X::DataFrame, Y::Array; equality=false)

    lnr.equality = equality

    # Convert dataframe to matrix 
    X = Matrix(X)
    
    # Initialize model 
    net = MLP(
        #[Dense(size(X, 2), 100),Dense(100, 100), Dense(100, 2, identity)],
        [Dense(size(X, 2), 100), Dense(100, 2, identity)],
        nll
    );
    lnr.mlp = net
    
    # Initialize training set 
    dtrn = minibatch(Float32.(X)', Int32.(1 .+ 1*(Y .>= 0)), 16)

    # Train for 100 epochs
    adam!(lnr.mlp, repeat(dtrn, 100));

end


"""
Used to make predictions based on a learned MLP regressor
"""
function predict(lnr::MLP_Classifier, X::DataFrame)
    
    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    X = Matrix(Matrix(X)')

    yy = lnr.mlp(X)

    return 1*(yy[2,:]-yy[1,:] .>= 0)
end


"""
Used to make predictions based on a learned MLP regressor
"""
function predict(lnr::MLP_Regressor, X::DataFrame)
    
    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end
    X = Matrix(X)
    # X = StatsBase.transform(lnr.scaler, X)

    X = Matrix(X')

    return lnr.mlp(X)'
end


"""
Evaluate using MLP in classification task
"""
function evaluate(lnr::MLP_Classifier, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = classification_evaluation()

    score = evaluator.second(y_pred, 1*(Y .>= 0))
    return score
end
"""
Evaluate using MLP in regression task
"""
function evaluate(lnr::MLP_Regressor, X::DataFrame, Y::Array)
    y_pred = predict(lnr, X)
    XX = Matrix(X)

    evaluator = regression_evaluation()

    score = evaluator.second(y_pred, Y)
    return score
end



"""
Embed MIO constraints on MLP classifier
"""
function embed_mio!(lnr::MLP_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)

    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    relax_var = gm.relax_coeff ==0 ? 0 : gm.relax_var;
    
    mlp = lnr.mlp
    m, x = gm.model, bbl.vars
    
    cons = []

    M_u, M_l = 1e5, -1e5

    layer_input = x

    # @TODO: probably use dependent var here?
    var_name = eval(Meta.parse(":y_nn_$(bbl.name)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    y = m[var_name];

    max_layers = length(mlp.layers)


    for (i, layer) in enumerate(mlp.layers)
        W, b = layer.w, layer.b
        if i == max_layers
            push!(cons,
                @constraint(m, y .== (W[2, :]'*layer_input + b[2])-(W[1, :]'*layer_input + b[1]))
            )
            
        else 
            v_pos_list = []
            
            for node_id in 1:size(W, 1)
                v = @variable(m, base_name="v_$(i)_$(node_id)")
                v_ind = @variable(m, binary=true, base_name="v_ind_$(i)_$(node_id)")
                push!(v_pos_list, v)

                append!(cons, [
                    @constraint(m, v >= 0),
                    @constraint(m, v >= W[node_id, :]'*layer_input + b[node_id]),
                    @constraint(m, v <= M_u*v_ind),
                    @constraint(m, v <= W[node_id, :]'*layer_input + b[node_id]-M_l*(1-v_ind))
                ])
                
            end
            layer_input = v_pos_list
        end
    end
    push!(cons, 
        @constraint(m, y + relax_var>= 0)
    )
    
    return Dict(1 => cons), Dict()
end



"""
Embed MIO constraints on MLP regressor
"""
function embed_mio!(lnr::MLP_Regressor, gm::GlobalModel, bbl::BlackBoxRegressor; kwargs...)

    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    relax_var = gm.relax_coeff ==0 ? 0 : gm.relax_var;

    mlp = lnr.mlp
    m, x = gm.model, bbl.vars
    
    cons = []

    M_u, M_l = 1e5, -1e5

    layer_input = x

    # @TODO: probably use dependent var here?
    #println(":y_nn_$(bbl.name)")
    var_name = eval(Meta.parse(":y_nn_$(bbl.name)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    y = m[var_name];

    max_layers = length(mlp.layers)


    for (i, layer) in enumerate(mlp.layers)
        W, b = layer.w, layer.b
        if i == max_layers
            push!(cons, 
                @constraint(m, y .== layer_input'*W[1, :] + b[1])
            )
        else 
            v_pos_list = []
            
            for node_id in 1:size(W, 1)
                v = @variable(m, base_name="v_$(i)_$(node_id)")
                v_ind = @variable(m, binary=true, base_name="v_ind_$(i)_$(node_id)")
                push!(v_pos_list, v)

                append!(cons, [
                    @constraint(m, v >= 0),
                    @constraint(m, v >= W[node_id, :]'*layer_input + b[node_id]),
                    @constraint(m, v <= M_u*v_ind),
                    @constraint(m, v <= W[node_id, :]'*layer_input + b[node_id]-M_l*(1-v_ind))
                ])
                
            end
            layer_input = v_pos_list
        end
    end

    linking_constr = nothing 

    if isnothing(lnr.dependent_var) 
        if lnr.equality
            linking_constr = @constraint(m, y>= -relax_var)
        else 
            linking_constr = @constraint(m, relax_var+EPSILON>= y >= -EPSILON-relax_var)
        end
    else
        if lnr.equality
            linking_constr = @constraint(m, y == lnr.dependent_var)
        else 
            linking_constr = @constraint(m, lnr.dependent_var >= y-relax_var)
        end
    end

    push!(cons, 
        linking_constr
    )
    
    return Dict(1 => cons), Dict()
end