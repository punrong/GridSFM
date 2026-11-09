# GridSFM AC-OPF verifier (loads, line ratings, generators, costs, voltage limits).
#
# Julia API:
#   initialize_case("/path/to/texas_model.json")
#   verify_scenario_json(JSON3.write(patch))
#
# The patch is a dictionary keyed by component IDs, e.g.:
#   Dict("load" => Dict("12" => Dict("pd" => 0.50, "qd" => 0.20)))
# Different sections ("load", "branch", "gen", "bus") may be combined.
# Unspecified fields keep their original values. Use PowerModels case units.
#
# Module name retained because existing Python notebooks import
# `jl.GridSFMLoadVerifier`; the API itself is mode-independent.

module GridSFMLoadVerifier

using PowerModels, Ipopt, JSON3
using JuMP: optimizer_with_attributes

export initialize_case, verify_scenario_json

const BASE_CASE = Ref{Any}(nothing)
const SOLVER = Ref{Any}(nothing)

const ALLOWED_FIELDS = Dict(
    "load"   => Set(["pd", "qd"]),
    "branch" => Set(["rate_a", "rate_b", "rate_c"]),
    "gen"    => Set(["gen_status", "cost", "model", "ncost"]),
    "bus"    => Set(["vmin", "vmax"]),
)

struct InvalidScenario <: Exception
    message::String
end
Base.showerror(io::IO, err::InvalidScenario) = print(io, err.message)
invalid(msg::AbstractString) = throw(InvalidScenario(String(msg)))

finite_or_nothing(v) = (v isa Real && !(v isa Bool) && isfinite(v)) ? Float64(v) : nothing

function require_finite(v, path::AbstractString)
    (v isa Real && !(v isa Bool)) || invalid("$path must be a number")
    f = Float64(v)
    isfinite(f) || invalid("$path must be finite")
    return f
end

function require_integer(v, path::AbstractString)
    (v isa Integer && !(v isa Bool)) || invalid("$path must be an integer")
    return Int(v)
end

function require_binary(v, path::AbstractString)
    i = require_integer(v, path)
    i in (0, 1) || invalid("$path must be 0 or 1")
    return i
end

function response_error(status::String, reason::String; stage::String="validation")
    return JSON3.write(Dict(
        "accepted" => false,
        "status" => status,
        "stage" => stage,
        "termination_status" => status == "SOLVER_ERROR" ? "ERROR" : "NOT_RUN",
        "primal_status" => "UNKNOWN",
        "objective" => nothing,
        "solve_time" => nothing,
        "reason" => reason,
        "applied_components" => Dict{String, Int}(),
        "modes" => String[],
        "warnings" => String[],
    ))
end

"""
    initialize_case(case_path)

Parse one PowerModels JSON network and prepare the Ipopt solver. Run once
before calling `verify_scenario_json` for any perturbation mode.
"""
function initialize_case(case_path::AbstractString)
    BASE_CASE[] = nothing
    SOLVER[] = nothing

    isfile(case_path) || error("Case JSON not found: $case_path")
    data = PowerModels.parse_file(String(case_path); import_all=false, validate=true)
    get(data, "multinetwork", false) == false ||
        error("This verifier expects one network, not a multinetwork case")

    optimizer = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes",
        "max_iter" => 3000, "tol" => 1e-6, "acceptable_tol" => 1e-4)
    PowerModels.silence()

    BASE_CASE[] = data
    SOLVER[] = optimizer
    return JSON3.write(Dict(
        "initialized" => true,
        "n_loads" => length(data["load"]),
        "n_buses" => length(data["bus"]),
        "n_generators" => length(data["gen"]),
        "n_branches" => length(data["branch"]),
        "per_unit" => get(data, "per_unit", nothing),
        "baseMVA" => get(data, "baseMVA", nothing),
        "supported_modes" => sort!(collect(keys(ALLOWED_FIELDS))),
    ))
end

# Apply one validated sparse update to a deep copy of BASE_CASE[].
function update_component!(component, original, table::String, id::String, patch)
    patch isa AbstractDict || invalid("$table.$id must be an object of field updates")
    isempty(patch) && invalid("$table.$id has no fields to update")
    for (raw_field, _) in pairs(patch)
        field = string(raw_field)
        field in ALLOWED_FIELDS[table] ||
            invalid("Unsupported field $table.$id.$field; allowed: $(join(sort!(collect(ALLOWED_FIELDS[table])), ", "))")
    end

    if table == "load"
        Int(get(original, "status", 1)) == 1 || invalid("load.$id is inactive in the original network")
        for field in ("pd", "qd")
            if haskey(patch, field)
                component[field] = require_finite(patch[field], "load.$id.$field")
            end
        end

    elseif table == "branch"
        Int(get(original, "br_status", 1)) == 1 || invalid("branch.$id is inactive in the original network")
        get(original, "transformer", false) == false ||
            invalid("branch.$id is a transformer; line derating here is restricted to non-transformer branches")
        for field in ("rate_a", "rate_b", "rate_c")
            if haskey(patch, field)
                haskey(original, field) || invalid("branch.$id does not have original $field")
                old_rate = require_finite(original[field], "original branch.$id.$field")
                old_rate > 0 || invalid("Original branch.$id.$field must be > 0 for derating")
                new_rate = require_finite(patch[field], "branch.$id.$field")
                new_rate > 0 || invalid("branch.$id.$field must be > 0")
                new_rate <= old_rate + 1e-10 * max(1.0, abs(old_rate)) ||
                    invalid("branch.$id.$field=$new_rate exceeds original rating $old_rate (not derating)")
                component[field] = new_rate
            end
        end

    elseif table == "gen"
        if haskey(patch, "gen_status")
            component["gen_status"] = require_binary(patch["gen_status"], "gen.$id.gen_status")
        end
        if haskey(patch, "cost")
            raw_cost = patch["cost"]
            raw_cost isa AbstractVector || invalid("gen.$id.cost must be an array of cost values")
            isempty(raw_cost) && invalid("gen.$id.cost cannot be empty")
            cost = [require_finite(x, "gen.$id.cost[$i]") for (i, x) in enumerate(raw_cost)]
            model = if haskey(patch, "model")
                require_integer(patch["model"], "gen.$id.model")
            elseif haskey(original, "model")
                require_integer(original["model"], "original gen.$id.model")
            else
                invalid("gen.$id has no original cost model; supply model=1 or model=2 with cost")
            end
            model in (1, 2) || invalid("gen.$id.model must be 1 (piecewise linear) or 2 (polynomial)")
            if model == 1
                length(cost) >= 4 && iseven(length(cost)) ||
                    invalid("gen.$id piecewise-linear cost must have at least two [power,cost] points")
                all(cost[i] < cost[i + 2] for i in 1:2:(length(cost) - 2)) ||
                    invalid("gen.$id piecewise-linear cost power breakpoints must increase")
            end
            ncost = model == 1 ? div(length(cost), 2) : length(cost)
            if haskey(patch, "ncost")
                requested = require_integer(patch["ncost"], "gen.$id.ncost")
                requested == ncost || invalid("gen.$id.ncost=$requested does not match cost array (expected $ncost)")
            end
            component["cost"] = cost
            component["model"] = model
            component["ncost"] = ncost
        elseif haskey(patch, "model") || haskey(patch, "ncost")
            invalid("gen.$id: model/ncost changes require a cost array in the same patch")
        end

    elseif table == "bus"
        Int(get(original, "bus_type", 1)) != 4 || invalid("bus.$id is inactive")
        vmin = haskey(patch, "vmin") ? require_finite(patch["vmin"], "bus.$id.vmin") : require_finite(original["vmin"], "original bus.$id.vmin")
        vmax = haskey(patch, "vmax") ? require_finite(patch["vmax"], "bus.$id.vmax") : require_finite(original["vmax"], "original bus.$id.vmax")
        vmin > 0 && vmax > 0 && vmin <= vmax || invalid("bus.$id needs 0 < vmin <= vmax")
        # Voltage 'squeeze' is tightening the original allowed range.
        original_min = require_finite(original["vmin"], "original bus.$id.vmin")
        original_max = require_finite(original["vmax"], "original bus.$id.vmax")
        vmin + 1e-10 >= original_min || invalid("bus.$id.vmin relaxes the original bound; expected a voltage squeeze")
        vmax <= original_max + 1e-10 || invalid("bus.$id.vmax relaxes the original bound; expected a voltage squeeze")
        component["vmin"] = vmin
        component["vmax"] = vmax
    end
    return nothing
end

"""
    verify_scenario_json(scenario_json)

Verify a SPARSE JSON object with optional `load`, `branch`, `gen`, and `bus`
sections. Each section maps *existing component IDs* to field updates.
You may combine modes in one scenario. Unspecified values stay unchanged.

All values must be in the units returned by `PowerModels.parse_file` (for
GridSFM `*_model.json` files, typically per-unit). No default physical
plausibility thresholds or Gaussian sampling rules are invented here.

Return JSON with `accepted`, `status`, solver statuses, `reason`, mode counts,
and optional warnings. A non-successful Ipopt run is "NOT_VERIFIED", which is
not automatically a mathematical proof of infeasibility.
"""
function verify_scenario_json(scenario_json::AbstractString)
    BASE_CASE[] === nothing && return response_error("NOT_INITIALIZED", "Call initialize_case first")
    local scenario
    try
        scenario = JSON3.read(String(scenario_json))
        scenario isa AbstractDict || invalid("Scenario must be a JSON object with load/branch/gen/bus sections")
    catch err
        return response_error("INVALID_SCENARIO", sprint(showerror, err))
    end

    # Validation is kept separate from the solve, to avoid labeling malformed
    # samples as "solver errors" or "physically infeasible" scenarios.
    data = deepcopy(BASE_CASE[])
    counts = Dict{String, Int}()
    warnings = String[]
    try
        isempty(scenario) && invalid("Scenario has no perturbation sections")
        for (raw_table, updates) in pairs(scenario)
            table = string(raw_table)
            haskey(ALLOWED_FIELDS, table) ||
                invalid("Unsupported section '$table'; use load, branch, gen, or bus")
            updates isa AbstractDict || invalid("'$table' section must be a JSON object keyed by component ID")
            isempty(updates) && invalid("'$table' section is empty")
            haskey(data, table) || invalid("Base case has no '$table' table")
            count = 0
            for (raw_id, patch) in pairs(updates)
                id = string(raw_id)
                haskey(data[table], id) || invalid("Unknown component ID $table.$id")
                update_component!(data[table][id], BASE_CASE[][table][id], table, id, patch)
                count += 1
                if table == "branch" && (haskey(patch, "rate_b") || haskey(patch, "rate_c"))
                    msg = "Standard PowerModels AC-OPF uses rate_a for thermal constraints; rate_b/rate_c may not affect feasibility"
                    msg in warnings || push!(warnings, msg)
                end
            end
            counts[table] = count
        end
    catch err
        return response_error("INVALID_SCENARIO", sprint(showerror, err))
    end

    try
        # Run AC polar OPF with the configured Ipopt solver.
        pm = PowerModels.instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
        result = PowerModels.optimize_model!(pm, optimizer=SOLVER[])
        termination = string(get(result, "termination_status", "UNKNOWN"))
        primal = string(get(result, "primal_status", "UNKNOWN"))
        # Accept only a successful solve with a feasible primal point.
        accepted = (termination in ("LOCALLY_SOLVED", "OPTIMAL")) && primal == "FEASIBLE_POINT"
        return JSON3.write(Dict(
            "accepted" => accepted,
            "status" => accepted ? "OPF_SOLVED" : "NOT_VERIFIED",
            "stage" => "opf",
            "termination_status" => termination,
            "primal_status" => primal,
            "objective" => finite_or_nothing(get(result, "objective", nothing)),
            "solve_time" => finite_or_nothing(get(result, "solve_time", nothing)),
            "reason" => accepted ? "" : "AC-OPF solution not verified: termination=$termination, primal=$primal",
            "applied_components" => counts,
            "modes" => sort!(collect(keys(counts))),
            "warnings" => warnings,
        ))
    catch err
        return JSON3.write(Dict(
            "accepted" => false,
            "status" => "SOLVER_ERROR",
            "stage" => "opf",
            "termination_status" => "ERROR",
            "primal_status" => "UNKNOWN",
            "objective" => nothing,
            "solve_time" => nothing,
            "reason" => sprint(showerror, err),
            "applied_components" => counts,
            "modes" => sort!(collect(keys(counts))),
            "warnings" => warnings,
        ))
    end
end

end # module GridSFMLoadVerifier