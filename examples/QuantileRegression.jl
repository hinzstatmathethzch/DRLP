include("../solvers/solverInverse.jl")


println("Performing Quantile Regression with the following parameters:")

n=100
X=rand(n,20)
Y=rand(n)
alpha=0.1

println("X=")
display(X)
println("\nY=")
display(Y)
println("\nalpha=$alpha")

include("../models/QREG.jl")
model=QREG_Model(X,Y,alpha)
x,path=solverInverse(model;sleepduration=0.0);


println("Found local minimum at x=$x with function value f(x)=$(f(model,x)[1])")
