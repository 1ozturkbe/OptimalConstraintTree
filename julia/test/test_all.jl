#=
all_tests:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
Note: All tests should be run from a julia REPR within the julia folder, using:
      julia --project
      include("test/test_all.jl")
      To see coverage, run with:
      julia --project --code-coverage=tracefile-%p.info --code-coverage=user
      include("test/test_all.jl")
=#


include("../src/OptimalConstraintTree.jl")
using Test

@testset "OptimalConstraintTree" begin
    @test 1+1 == 2

    include("test_src.jl")

    include("test_tools.jl")

    include("test/test_bbf.jl")

end

# Other tests to try later

# include("test/test_transonic.jl");