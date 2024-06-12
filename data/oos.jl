using Parameters, Random

@with_kw mutable struct oos_params
    g = 9.81
    mu = 3.986e14 # gravitational constant
    rE = 6378137 # radius of Earth (m)
    dry_mass = 500 # empty mass of satellite((kg))
    Isp = 230     # Specific impulse
    alt_sat = 780e3 # altitude of satellites (m)
    n_sats = 7     # number of satellites (must be odd for this case)
    true_anomalies = circshift(collect(1:n_sats).*2*pi/n_sats .- 2pi*6/n_sats, Int((n_sats-1)/2)+1)
    fuel_required = circshift(collect(100 .+ range(-20, stop=40, length=n_sats)), 4) + rand(MersenneTwister(314), n_sats) .* 80 # dummy example 
    r_sat = rE + alt_sat
    period_sat = 2pi*sqrt(r_sat^3/mu)
    r_orbits = rE .+ 1e3 .* collect(760:1:800)
    orbital_periods = [2pi*sqrt(r^3/mu) for r in r_orbits]
    transfer_time = [2*pi*sqrt((r_sat + r)^3/(8*mu)) for r in r_orbits]
    maximum_time = n_sats * 0.05*3600*24*365 # 0.35 years in seconds
end

function oos_gm!(op = oos_params())
    n = op.n_sats

    # Making sure the orbits don't include the actual orbit of the satellite
    op.r_orbits = filter!(r -> r != op.r_sat, op.r_orbits)  
    op.orbital_periods = [2pi*sqrt(r^3/op.mu) for r in op.r_orbits]
    op.transfer_time = [2*pi*sqrt((op.r_sat + r)^3/(8*op.mu)) for r in op.r_orbits]

    # Starting a JuMP.Model
    m = Model(SOLVER_SILENT)

    @variable(m, 1.0025 >= dmass_entry[i=1:n-1] >= 1)
    @variable(m, 1.0025 >= dmass_exit[i=1:n-1] >= 1)
    # Locations xx, and sat_order (mostly for post_processing)
    @variable(m, n >= sat_order[i=1:n] >= 1)
    @variable(m, xx[1:n, 1:n], Bin)        # transfer occurs from row index to column index
    set_lower_bound.(xx, 0)
    set_upper_bound.(xx, 1)

    # Maneuver time variables
    @variable(m, 500 >= N_orbit[1:n-1] >= 50) # Artificial lower and upper bounds.
    @variable(m,  maximum(abs.(op.period_sat .- op.orbital_periods)) >= dt_orbit[1:n-1] >= -maximum(abs.(op.period_sat .- op.orbital_periods)))
    @variable(m, maximum(op.transfer_time) >= dt_transfer_orbit[1:n-1] >= minimum(op.transfer_time))
    @variable(m, maximum(op.r_orbits) >= r_orbit[1:n-1] >= minimum(op.r_orbits))
    @variable(m, maximum(op.orbital_periods) >= period_orbit[1:n-1] >= minimum(op.orbital_periods))

    # Program can pick the best starting satellite as well. 
    @constraint(m, [i=1:n], sat_order[i] == sum(xx[i, :] .* collect(1:n)))
    @constraint(m, [i=1:n], sum(xx[i, :]) == 1) # Exactly one transfer every time (same as each satellite visited once)
    @constraint(m, [i=1:n], sum(xx[:, i]) == 1)   # At most one transfer to each satellite

    # # True anomaly computation from satellite order
    @variable(m, pi >= ta[1:n-1] >= -pi)
    @constraint(m, [i=1:n-1], ta[i] == sum(op.true_anomalies .* xx[i+1,:]) - 
                                        sum(op.true_anomalies .* xx[i,:]))

    # Fuel consumption (Watch these arbitrary upper bounds)
    @variable(m, 2000 >= masses[1:n-1, 1:5] >= op.dry_mass)
    @variable(m, 2000 >= wet_mass >= op.dry_mass)

    # Orbit choice, and time and fuel cost
    @variable(m, 1.0025 >= fractional_dmasses[1:n-1, j=1:4] >= 1) # reduction in masses in maneuvers
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 1] == dmass_entry[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 2] == dmass_exit[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 3] == dmass_exit[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 4] == dmass_entry[i])

    # Computing required fuel depending on transfer schedule
    @variable(m, maximum(op.fuel_required) >= fuel_needed[1:n] >= minimum(op.fuel_required))
    @constraint(m, [i=1:n], fuel_needed[i] == sum(xx[i, :] .* op.fuel_required))

    # Mass conservation constraints (linear)
    @constraint(m, wet_mass == masses[1, 1] + fuel_needed[1])
    for i=1:n-1
        @constraint(m, masses[i, 1] >= masses[i, 2])
        @constraint(m, masses[i, 2] >= masses[i, 3])
        @constraint(m, masses[i, 3] >= masses[i, 4])
        @constraint(m, masses[i, 4] >= masses[i, 5])
    end
    @constraint(m, [i=1:n-2], masses[i, 5] == masses[i+1, 1] + fuel_needed[i+1])
    @constraint(m, masses[n-1, 5] == op.dry_mass + fuel_needed[n])

    # Maneuver time constraints (linear)
    @constraint(m, [i=1:n-1], dt_orbit[i] == period_orbit[i] - op.period_sat)
    @variable(m, 2 * op.maximum_time / op.n_sats >= t_maneuver[1:n-1] >= 0)
    @constraint(m, sum(t_maneuver) <= op.maximum_time)

    @objective(m, Min, wet_mass)

    gm = GlobalModel(model = m, name = "oos")

    add_nonlinear_constraint(gm, :((r_orbit) -> maximum([exp(1/($(op.g)*$(op.Isp)) * 
                                sqrt($(op.mu)/r_orbit[1])*(sqrt(2 * $(op.r_sat)/($(op.r_sat) + r_orbit[1]))-1)), 
                                                                      exp(1/($(op.g)*$(op.Isp)) * 
                                sqrt($(op.mu)/$(op.r_sat))*(sqrt(2 * r_orbit[1]/($(op.r_sat) + r_orbit[1]))-1))])), 
                                vars = [r_orbit[1]], dependent_var = dmass_entry[1],
                                name = "dmass_entry", equality = true)
    add_nonlinear_constraint(gm, :((r_orbit) -> maximum([exp(1/($(op.g)*$(op.Isp)) * 
                                sqrt($(op.mu)/$(op.r_sat))*(1 - sqrt(2 * r_orbit[1]/($(op.r_sat) + r_orbit[1])))),
                                                                      exp(1/($(op.g)*$(op.Isp)) * 
                                sqrt($(op.mu)/r_orbit[1])*(1 - sqrt(2 * $(op.r_sat)/($(op.r_sat) + r_orbit[1]))))])), 
                                vars = [r_orbit[1]], dependent_var = dmass_exit[1],
                                name = "dmass_exit", equality = true)
    for i = 2:n-1
        add_linked_constraint(gm, gm.bbls[end-1], [r_orbit[i]], dmass_entry[i])
        add_linked_constraint(gm, gm.bbls[end], [r_orbit[i]], dmass_exit[i])
    end    
    
    # Mass conservation constraints (bilinear)
    add_nonlinear_constraint(gm, :((x,y) -> x*y), 
        vars = [masses[1, 2], fractional_dmasses[1,1]], 
        dependent_var = masses[1,1],
        expr_vars = [masses[1, 2], fractional_dmasses[1,1]],
        name = "mass_fraction", equality = true)
    for j=2:4
        add_linked_constraint(gm, gm.bbls[end], [masses[1, j+1], fractional_dmasses[1,j]], masses[1,j])
    end
    for i=2:n-1
        for j=1:4
            add_linked_constraint(gm, gm.bbls[end], [masses[i, j+1], fractional_dmasses[i,j]], masses[i,j])
        end
    end

    # Orbital period constraint (equality)
    add_nonlinear_constraint(gm, :(r_orbit -> 2*pi*sqrt(r_orbit[1]^3/$(op.mu))), 
        vars = [r_orbit[1]], dependent_var = period_orbit[1],
        name = "orbital_period", equality=true)
    [add_linked_constraint(gm, gm.bbls[end], [r_orbit[i]], period_orbit[i]) for i=2:n-1]

    # Transfer time constraint
    add_nonlinear_constraint(gm, :(r_orbit -> 2*pi*sqrt(($(op.r_sat) + r_orbit[1])^3/(8*$(op.mu)))), 
        vars = [r_orbit[1]], dependent_var = dt_transfer_orbit[1],
        name = "transfer_time", equality = true)
    [add_linked_constraint(gm, gm.bbls[end], [r_orbit[i]], dt_transfer_orbit[i]) for i=2:n-1]

    # Orbital revolutions constraint (equality)
    add_nonlinear_constraint(gm, :((dt_orbit, N_orbit) -> N_orbit[1]*dt_orbit[1]/ $(op.period_sat)), 
        vars = [dt_orbit[1], N_orbit[1]], dependent_var = ta[1],
        name = "orbital_revolutions", equality=true)
    [add_linked_constraint(gm, gm.bbls[end], [dt_orbit[i], N_orbit[i]], ta[i]) for i=2:n-1]

    # Maneuver time constraint
    add_nonlinear_constraint(gm, :((dt_transfer_orbit, N_orbit, period_orbit) -> 
            (dt_transfer_orbit[1] + N_orbit[1] * period_orbit[1])), 
            vars = [dt_transfer_orbit[1], N_orbit[1], period_orbit[1]], dependent_var = t_maneuver[1],
            name = "maneuver_time", equality = true)
    [add_linked_constraint(gm, gm.bbls[end], [dt_transfer_orbit[i], N_orbit[i], period_orbit[i]], t_maneuver[i])
        for i=2:n-1]

    set_param(gm, :ignore_accuracy, true)

    return gm
end

function oos_baron!(op = oos_params())
    n = op.n_sats

    # Making sure the orbits don't include the actual orbit of the satellite
    op.r_orbits = filter!(r -> r != op.r_sat, op.r_orbits)  
    op.orbital_periods = [2pi*sqrt(r^3/op.mu) for r in op.r_orbits]
    op.transfer_time = [2*pi*sqrt((op.r_sat + r)^3/(8*op.mu)) for r in op.r_orbits]

    # Starting a JuMP.Model
    m = Model(SOLVER_SILENT)

    @variable(m, 1.0025 >= dmass_entry[i=1:n-1] >= 1)
    @variable(m, 1.0025 >= dmass_exit[i=1:n-1] >= 1)
    # Locations xx, and sat_order (mostly for post_processing)
    @variable(m, n >= sat_order[i=1:n] >= 1)
    @variable(m, xx[1:n, 1:n], Bin)        # transfer occurs from row index to column index
    set_lower_bound.(xx, 0)
    set_upper_bound.(xx, 1)

    # Maneuver time variables
    @variable(m, 500 >= N_orbit[1:n-1] >= 50) # Artificial lower and upper bounds.
    @variable(m,  maximum(abs.(op.period_sat .- op.orbital_periods)) >= dt_orbit[1:n-1] >= -maximum(abs.(op.period_sat .- op.orbital_periods)))
    @variable(m, maximum(op.transfer_time) >= dt_transfer_orbit[1:n-1] >= minimum(op.transfer_time))
    @variable(m, maximum(op.r_orbits) >= r_orbit[1:n-1] >= minimum(op.r_orbits))
    @variable(m, maximum(op.orbital_periods) >= period_orbit[1:n-1] >= minimum(op.orbital_periods))

    # Program can pick the best starting satellite as well. 
    @constraint(m, [i=1:n], sat_order[i] == sum(xx[i, :] .* collect(1:n)))
    @constraint(m, [i=1:n], sum(xx[i, :]) == 1) # Exactly one transfer every time (same as each satellite visited once)
    @constraint(m, [i=1:n], sum(xx[:, i]) == 1)   # At most one transfer to each satellite

    # # True anomaly computation from satellite order
    @variable(m, pi >= ta[1:n-1] >= -pi)
    @constraint(m, [i=1:n-1], ta[i] == sum(op.true_anomalies .* xx[i+1,:]) - 
                                        sum(op.true_anomalies .* xx[i,:]))

    # Fuel consumption (Watch these arbitrary upper bounds)
    @variable(m, 2000 >= masses[1:n-1, 1:5] >= op.dry_mass)
    @variable(m, 2000 >= wet_mass >= op.dry_mass)

    # Orbit choice, and time and fuel cost
    @variable(m, 1.0025 >= fractional_dmasses[1:n-1, j=1:4] >= 1) # reduction in masses in maneuvers
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 1] == dmass_entry[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 2] == dmass_exit[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 3] == dmass_exit[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 4] == dmass_entry[i])

    # Computing required fuel depending on transfer schedule
    @variable(m, maximum(op.fuel_required) >= fuel_needed[1:n] >= minimum(op.fuel_required))
    @constraint(m, [i=1:n], fuel_needed[i] == sum(xx[i, :] .* op.fuel_required))

    # Mass conservation constraints (linear)
    @constraint(m, wet_mass == masses[1, 1] + fuel_needed[1])
    for i=1:n-1
        @constraint(m, masses[i, 1] >= masses[i, 2])
        @constraint(m, masses[i, 2] >= masses[i, 3])
        @constraint(m, masses[i, 3] >= masses[i, 4])
        @constraint(m, masses[i, 4] >= masses[i, 5])
    end
    @constraint(m, [i=1:n-2], masses[i, 5] == masses[i+1, 1] + fuel_needed[i+1])
    @constraint(m, masses[n-1, 5] == op.dry_mass + fuel_needed[n])

    # Maneuver time constraints (linear)
    @constraint(m, [i=1:n-1], dt_orbit[i] == period_orbit[i] - op.period_sat)
    @variable(m, 2 * op.maximum_time / op.n_sats >= t_maneuver[1:n-1] >= 0)
    @constraint(m, sum(t_maneuver) <= op.maximum_time)

    @objective(m, Min, wet_mass)
    
    @variable(m, dmass_entry_bin[i=1:n-1, j=1:2], Bin)
    @constraint(m, [i=1:n-1], sum(dmass_entry_bin[i, :]) == 1)
    @variable(m, dmass_exit_bin[i=1:n-1, j=1:2], Bin)
    @constraint(m, [i=1:n-1], sum(dmass_exit_bin[i, :]) == 1)

    # Upper bounding dmass_entry (big-M)
    @NLconstraint(m, [i = 1:n-1], dmass_entry[i] <= 2*(1-dmass_entry_bin[i, 1]) + exp(1/(op.g*op.Isp) * (op.mu*r_orbit[i]^(-1))^0.5*((2 * op.r_sat*(op.r_sat + r_orbit[i])^(-1))^0.5-1)))
    @NLconstraint(m, [i = 1:n-1], dmass_entry[i] <= 2*(1-dmass_entry_bin[i, 2]) + exp(1/(op.g*op.Isp) * (op.mu*op.r_sat^(-1))^0.5*((2 * r_orbit[i]*(op.r_sat + r_orbit[i])^(-1))^0.5-1)))

    # Lower bounding dmass_entry
    @NLconstraint(m, [i = 1:n-1], dmass_entry[i] >= exp(1/(op.g*op.Isp) * (op.mu*r_orbit[i]^(-1))^0.5*((2 * op.r_sat*(op.r_sat + r_orbit[i])^(-1))^0.5-1)))
    @NLconstraint(m, [i = 1:n-1], dmass_entry[i] >= exp(1/(op.g*op.Isp) * (op.mu/op.r_sat)^0.5*((2 * r_orbit[i]*(op.r_sat + r_orbit[i])^(-1))^0.5-1)))

    # Upper bounding dmass_exit (big-M)
    @NLconstraint(m, [i = 1:n-1], dmass_exit[i] <= 2*(1 - dmass_exit_bin[i,1]) + exp(1/(op.g*op.Isp) * (op.mu*op.r_sat^(-1))^0.5*(1 - (2 * r_orbit[i]*(op.r_sat + r_orbit[i])^(-1))^0.5)))
    @NLconstraint(m, [i = 1:n-1], dmass_exit[i] <= 2*(1 - dmass_exit_bin[i,2]) + exp(1/(op.g*op.Isp) * (op.mu*r_orbit[i]^(-1))^0.5*(1 - (2 * op.r_sat*(op.r_sat + r_orbit[i])^(-1))^0.5)))

    # Lower bounding dmass_exit
    @NLconstraint(m, [i = 1:n-1], dmass_exit[i] >= exp(1/(op.g*op.Isp) * (op.mu/op.r_sat)^0.5*(1 - (2 * r_orbit[i]*(op.r_sat + r_orbit[i])^(-1))^(0.5))))
    @NLconstraint(m, [i = 1:n-1], dmass_exit[i] >= exp(1/(op.g*op.Isp) * (op.mu*r_orbit[i]^(-1))^0.5*(1 - (2 * op.r_sat*(op.r_sat + r_orbit[i])^(-1))^0.5)))

    # Mass conservation constraints (bilinear)
    @NLconstraint(m, [i = 1:n-1, j = 1:4], masses[i,j] == masses[i, j+1] * fractional_dmasses[i,j])

    # Orbital period constraint (equality)
    @NLconstraint(m, [i = 1:n-1], period_orbit[i] == 2*pi*(r_orbit[i]^3/op.mu)^0.5)

    # Transfer time constraint
    @NLconstraint(m, [i = 1:n-1], dt_transfer_orbit[i] == 2*pi*((op.r_sat + r_orbit[i])^3/(8*op.mu))^0.5)

    # Orbital revolutions constraint (equality)
    @NLconstraint(m, [i = 1:n-1], ta[i] == N_orbit[i]*dt_orbit[i]/op.period_sat)

    # Maneuver time constraint
    @NLconstraint(m, [i = 1:n-1], t_maneuver[i] == dt_transfer_orbit[i] + N_orbit[i] * period_orbit[i])

    return m
end

# m = oos_baron!()
# set_optimizer(m, BARON.Optimizer)
# optimize!(m)