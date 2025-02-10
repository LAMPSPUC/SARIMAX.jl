using Test

@testset "Sarimax Exception Tests" begin
    @test_throws ModelNotFitted begin
        # Code that should throw ModelNotFitted
        try
            throw(ModelNotFitted())
        catch e
            println(e)
            @test e isa ModelNotFitted
            throw(ModelNotFitted())
        end
    end

    @test_throws MissingMethodImplementation begin
        # Code that should throw MissingMethodImplementation
        try
            throw(MissingMethodImplementation("method"))
        catch e
            println(e)
            @test e isa MissingMethodImplementation
            @test e.method == "method"
            throw(MissingMethodImplementation("method"))
        end
    end

    @test_throws InconsistentDatePattern begin
        # Code that should throw InconsistentDatePattern
        try
            throw(InconsistentDatePattern())
        catch e
            println(e)
            @test e isa InconsistentDatePattern
            throw(InconsistentDatePattern())
        end
    end

    @test_throws MissingExogenousData begin
        # Code that should throw MissingExogenousData
        try
            throw(MissingExogenousData())
        catch e
            println(e)
            @test e isa MissingExogenousData
            throw(MissingExogenousData())
        end
    end

    @test_throws InvalidParametersCombination begin
        # Code that should throw InvalidParametersCombination
        try
            throw(InvalidParametersCombination("msg"))
        catch e
            println(e)
            @test e isa InvalidParametersCombination
            @test e.msg == "msg"
            throw(InvalidParametersCombination("msg"))
        end
    end

end
