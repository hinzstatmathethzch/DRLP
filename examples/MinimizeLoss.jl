include("../solvers/solverInverse.jl")

X=rand(20,4)
Y=rand(20)

include("../models/LAD.jl")
model=LAD_Model(X,Y)

include("../models/QREG.jl")
model=QREG_Model(X,Y,0.1)

x,path=solverInverse(model;sleepduration=0.0);
x
println("Found local minimum at x=$x with function value f(x)=$(f(model,x)[1])")
