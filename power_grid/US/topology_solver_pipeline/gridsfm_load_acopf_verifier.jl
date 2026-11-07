# GridSFM load AC-OPF verifier — same PowerModels ACP + Ipopt workflow
# as gen_perturbed_data.jl. Imported once by Python via juliacall.
# No PYPOWER / MATPOWER conversion is involved.
module GridSFMLoadVerifier

using PowerModels, Ipopt, JuMP, JSON3
using JuMP: optimizer_with_attributes

const BASE_CASE = Ref{Any}(nothing)
const ACTIVE_IDS = Ref{Vector{String}}(String[])
const SOLVER = Ref{Any}(nothing)

finite_or_nothing(v) = (v isa Real && isfinite(v)) ? Float64(v) : nothing

function initialize_case(case_path::AbstractString, ids_json::AbstractString)
    isfile(case_path) || error("Case JSON not found: $case_path")
    # Exactly the parsing path used in gen_perturbed_data.jl
    data = PowerModels.parse_file(String(case_path); import_all=false, validate=true)
    ids = String.(JSON3.read(String(ids_json)))
    isempty(ids) && error("No active load IDs were provided")
    length(ids) == length(unique(ids)) || error("Repeated load IDs")
    for id in ids
        haskey(data["load"], id) || error("Load ID not in Julia case: $id")
        Int(get(data["load"][id], "status", 1)) == 1 || error("Inactive load ID: $id")
    end
    BASE_CASE[] = data
    ACTIVE_IDS[] = ids
    SOLVER[] = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes",
        "max_iter" => 3000, "tol" => 1e-6, "acceptable_tol" => 1e-4)
    PowerModels.silence()
    return JSON3.write(Dict("initialized" => true,
        "n_loads" => length(ids), "n_buses" => length(data["bus"]),
        "n_generators" => length(data["gen"]),
        "n_branches" => length(data["branch"])))
end

function verify_loads_json(load_values_json::AbstractString)
    try
        BASE_CASE[] === nothing && error("Call initialize_case first")
        pairs = JSON3.read(String(load_values_json))
        ids = ACTIVE_IDS[]
        length(pairs) == length(ids) || error("Expected $(length(ids)) Pd/Qd pairs, got $(length(pairs))")
        # Original network is preserved: only chosen active loads' Pd/Qd change.
        data = deepcopy(BASE_CASE[])
        for (id, pair) in zip(ids, pairs)
            length(pair) == 2 || error("Load $id must contain [Pd, Qd]")
            pd, qd = Float64(pair[1]), Float64(pair[2])
            isfinite(pd) && isfinite(qd) || error("Non-finite Pd/Qd for load $id")
            data["load"][id]["pd"] = pd
            data["load"][id]["qd"] = qd
        end

        # Same model and solver configuration as provided GridSFM Julia file.
        pm = PowerModels.instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
        result = PowerModels.optimize_model!(pm, optimizer=SOLVER[])

        termination = string(get(result, "termination_status", "UNKNOWN"))
        primal = string(get(result, "primal_status", "UNKNOWN"))
        # Strict acceptance: feasible point AND a locally solved/optimal status.
        accepted = termination in ("LOCALLY_SOLVED", "OPTIMAL") && primal == "FEASIBLE_POINT"
        return JSON3.write(Dict(
            "accepted" => accepted,
            "status" => accepted ? "OPF_SOLVED" : "NOT_VERIFIED",
            "termination_status" => termination,
            "primal_status" => primal,
            "objective" => finite_or_nothing(get(result, "objective", nothing)),
            "solve_time" => finite_or_nothing(get(result, "solve_time", nothing)),
            "reason" => accepted ? "" : "AC-OPF not accepted: termination=$termination, primal=$primal"
        ))
    catch err
        return JSON3.write(Dict(
            "accepted" => false,
            "status" => "SOLVER_ERROR",
            "termination_status" => "ERROR",
            "primal_status" => "UNKNOWN",
            "objective" => nothing,
            "solve_time" => nothing,
            "reason" => sprint(showerror, err)
        ))
    end
end

end # module
