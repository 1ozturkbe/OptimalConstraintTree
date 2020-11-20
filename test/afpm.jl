include("../test/load.jl")

using DataFrames
using Test
using CSV
using Gurobi

@info "Trains over axial flux motor data."

# Getting simulation data
X = CSV.read("data/afpm/afpm_inputs.csv",
                 copycols=true, delim=",");
X_infeas = CSV.read("data/afpm/afpm_infeas_inputs.csv",
                    copycols=true, delim=",");
Y = CSV.read("data/afpm/afpm_outputs.csv",
                 copycols=true, delim=",");
X = select!(X, Not(:Column1));
X_infeas = select!(X_infeas, Not(:Column1));
Y = select!(Y, Not(:Column1));

# Free variables
varkeys = Symbol.(names(X))
objkeys = Symbol.(names(Y))

# Creating model with relevant variables
m = Model(Gurobi.Optimizer)
inputs = [];
for key in varkeys
    push!(inputs, @variable(m, base_name = string(key)));
end
outputs = [];
for key in objkeys
    push!(outputs, @variable(m, base_name = string(key)));
end

# Bounding and Integer constraints for input variables
N_coils, TPC, p = inputs[4:6];
# baseline = Dict(inputs[i] => [13, 7.6, 1., 18, 10, 16, 0.15, 3.0, 0.45][i] for i=1:length(varkeys));
# ranges = Dict(key => [0.5*value, 1.5*value] for (key, value) in baseline);
# ranges[inputs[9]] = [0.25*baseline[inputs[9]], 2.25*baseline[inputs[9]]]; # wire_A needs special care...
# ranges = Dict(key => log.(value) for (key, value) in ranges)
ranges = Dict(var => [log.(minimum(X[string(var)])),log.(maximum(X[string(var)]))] for var in inputs)

# Bounding for output variables of interest
idxs = [1, 4, 6, 8, 10, 12];
feas_idxs = findall(x -> x .>= 0.5, Y.Efficiency);
X_feas = X[feas_idxs, :];
Y_feas = Y[feas_idxs, :];
out_ranges = Dict(key => [minimum((Y_feas[Symbol(key)])), maximum((Y_feas[Symbol(key)]))] for key in outputs[idxs]);
bound!(m, ranges);

# Geometry feasibility
# lnr = IAI.fit(base_otc(), )

# Geometry constraints (in logspace)
D_out, D_in, N_coils, wire_w = inputs[1], inputs[2], inputs[4], inputs[7]
@constraint(m, D_out >= D_in)
@constraint(m, log(pi) + D_in >= log(0.2) + wire_w + N_coils) #pi*D_in >= 2*0.1*wire_w*N_coils

N_coils_range = log.(unique(X_feas["N_coils"]));
int = @variable(m, [1:length(N_coils_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, N_coils == sum(N_coils_range .* int))

p_range = log.(unique(X_feas["p"]))
int = @variable(m, [1:length(p_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, p == sum(p_range .* int))
@constraint(m, N_coils >= p + 1e-3) # motor type 2

TPC_range = log.(unique(X_feas["TPC"]))
int = @variable(m, [1:length(TPC_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, TPC == sum(TPC_range .* int))

# Objectives and FOMs
output_idxs = [1, 4, 6, 8, 10, 12];
P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [outputs[idx] for idx in idxs]
set_upper_bound(Efficiency, 1)
@constraint(m, log(10.) == P_shaft)
@constraint(m, log.(8000) == Rotational_Speed)

# Fitting power closure, and creating a global model
# simulation = DataConstraint(vars = inputs)
feasmap = zeros(size(Y, 1)); feasmap[feas_idxs] .= 1;
# add_data!(simulation, log.(X), feasmap)
# learn_constraint!(simulation)
# lnr = IAI.fit!(base_otc(), log.(X), feasmap)
# IAI.write_json("power_closure.json", lnr)
lnr = IAI.read_json("power_closure.json")
# constrs, leaf_vars = add_feas_constraints!(m, inputs, lnr, M=1e3, return_data = true)
# simulation.mi_constraints = constrs; # Note: this is needed to monitor the presence of tree
# simulation.leaf_variables = leaf_vars; #  constraints and variables in gm.model

# Fitting appropriate power
feasmap = 10.2 .>= Y_feas["P_shaft"] .>= 9.8;
lnr = IAI.fit!(base_otc(), log.(X_feas), feasmap);
constrs, leaf_vars = add_feas_constraints!(m, inputs, lnr, M=1e3, return_data = true);

# Fitting appropriate RPM
feasmap = Y_feas["Rotational Speed"] .>= 7800;
lnr = IAI.fit!(base_otc(), log.(X_feas), feasmap);
constrs, leaf_vars = add_feas_constraints!(m, inputs, lnr, M=1e3, return_data = true)


# Only one feasible leaf (4), so can just use regression equations
# reg_lnr = base_otr()
# reg_lnr.hyperplane_config = (sparsity = 1,)
# for FOM in FOMs
#     lnr = IAI.fit!(reg_lnr, log.(X_feas), log.(Y_feas[string(FOM)]))
#     IAI.write_json(string(FOM) * ".json", lnr)
# end

leaf = 5;

# Regressions over leaves
# P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [outputs[idx] for idx in idxs]
# FOMs = [P_shaft, Rotational_Speed]
# leaf_index, all_leaves = bin_to_leaves(lnr, log.(X_feas))
# for FOM in FOMs
#     regressor = regress(log.(X_feas), log.(Y_feas[string(FOM)]))
#     constant = IAI.get_prediction_constant(regressor)
#     weights  = IAI.get_prediction_weights(regressor)[1]
#     vks = Symbol.(names(X_feas))
#     β = []
#     for i = 1:size(vks, 1)
#         if vks[i] in keys(weights)
#             append!(β, weights[vks[i]])
#         else
#             append!(β, 0.0)
#         end
#     end
#     @constraint(m, sum(β .* inputs) + constant == FOM)
# end

# Solving
@objective(m, Min, Mass)
optimize!(m)
println("Inputs")
for i=1:length(inputs)
    println(string(inputs[i], " ", exp(getvalue(inputs[i]))))
end
println("FOMs")
for FOM in FOMs
    println(string(FOM, " ", exp(getvalue(FOM))))
end

# fdf = DataFrame(names(X) .=> exp.(getvalue.(inputs)))
# CSV.write("data/afpm/afpm_opt.csv", fdf)