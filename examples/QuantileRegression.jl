include("../solver/solver.jl")

# set number of observations
n=10000

# sample training data
dim=4
X=rand(n,dim)
Y=rand(n)

# create Quantile Regression model
include("../models/QREG.jl")
alpha=0.8
model=QREG_Model(X,Y,alpha)

# set start point
x0=zeros(model.n0)

# train
println("Performing Quantile Regression with alpha=$alpha in dimension $dim")
(state,path,x)=solverLP(model,x0);
println("Found local minimum at x=")
display(x)
println("\nwith function value f(x)=$(f(model,x)[1])")

# optionally plot the path of scores:
# using Plots
# plot(last.(path))
