include("../solverLP/solverLP.jl")

# set number of observations
n=10000

# sample training data
dim=4
X=rand(n,dim)
Y=rand(n)

# create Least Absolute Deviation model
include("../models/LAD.jl")
model=LAD_Model(X,Y)

# set start point
x0=zeros(model.n0)

# train
println("Performing LAD Regression in dimension $dim")
(state,path,x)=solverLP(model,x0);
println("Found local minimum at x=")
display(x)
println("\nwith function value f(x)=$(f(model,x)[1])")
