include("../models/random.jl")
include("../models/L1DataModel.jl")
include("../solverLP/solverLP.jl")


randmodel=randomModel(4,[5,4,2])
n=500
# input parameters
Ydata=rand(Uniform(-3,3),n)
Xdata=rand(Uniform(-3,3),randmodel.n0,n)

l=1
datamodel=L1DataModel(randmodel,Xdata,Ydata,l)
model=datamodel

x0=rand(Uniform(-20,20),model.n0)
x=x0

(code,trace1,x)=solverLP(model,x);
code





