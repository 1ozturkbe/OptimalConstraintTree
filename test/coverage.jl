#=
test_coverage:
- Julia version: 
- Author: Berk
- Date: 2020-07-24
=#

# Get coverage (from within julia)
# Restart julia from within the julia folder and run this file.
global PROJECT_ROOT = @__DIR__
using Coverage
# process '*.cov' files
coverage = process_folder() # defaults to src/; alternatively, supply the folder name as argument
# process '*.info' files
coverage = merge_coverage_counts(coverage, filter!(
    let prefixes = (joinpath(pwd(), "src", ""))
        c -> any(p -> startswith(c.filename, p), prefixes)
    end,
    LCOV.readfolder("test")))
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
@show get_summary(process_file(joinpath("src", "OptimalConstraintTree.jl")))
println("Covered lines: ", covered_lines)
println("Total lines: ", total_lines)
println("Ratio: ", covered_lines/total_lines)
# To clear coverage files
# Coverage.