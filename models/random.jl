include("model.jl")
using Distributions
####################
# Function to generate random network 
# parameters for a given architecture
####################
function randomParameters(n0,n,L)
W=Array{Matrix{Float64},1}(undef,L+1)
b=Array{Array{Float64,1},1}(undef,L+1)
W[1]=rand(Uniform(-1,1),n[1],n0)
b[1]=rand(Uniform(-1,1),n[1])
for i = 2:L
	W[i]= rand(Uniform(-1,1),n[i],n[i-1])
	b[i]=rand(Uniform(-1,1),n[i])
end
W[L+1]=rand(Uniform(-1,1),1,n[L])
b[L+1]=rand(Uniform(-1,1),1)
return (W,b)
end

function randomModel(n0,n::Array{Int,1})
	L=size(n,1)
	(W,b)=randomParameters(n0,n,L)
	return Model(n0,n,n,L,W,b)
end

