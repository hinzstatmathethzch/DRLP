# the model is expected to be given in the following format:
# n0: input dimension
# n: width vector for the layers
# nNoDuplicate: width vector for layers that have no layer-wise duplicates in weight and bias
# L: number of hidden layers
# W: Array of weight matrices
# b: Array of bias vectors
#
# Additional restrictions:
# In every layer `l`, the model shall be arranged in such a way that from neuron index
# 1 up to neuron index `nNoDuplicate[l]` there shall be no negated or non-negated duplicates in the sense that for 
# `m1` and `m2` in `1:noDuplicate[l]` only for `m1==m2` it can happen that 
# `(W[l][m1,:]==W[l][m2,:] && b[l][m1]==b[l][m2])||(W[l][m1,:]==-W[l][m2,:] && b[l][m1]==-b[l][m2])`. After that index `nNoDuplicate[l]` only (negated) duplicates of weights and bias combinations occur. Furthermore they shall be ordered and the non-duplicates shall come first. More precisely for index m  from `1` to `n[l]-nNoDuplicate[l]` there shall be weight-bias combinations `(W[l][m1,:],b[l][m])` that do not occur for a different index in that layer, for index m from `n[l]-nNoDuplicate[l]` to `nNoDuplicate[l]` it shall hold that 
# `W[l][m,:]==-W[l][m+nNoDuplicate[l]-n[l],:]` and `b[l][m]==-b[l][m+nNoDuplicate[l]-n[l]]`
# or `W[l][m,:]==W[l][m+nNoDuplicate[l]-n[l],:]` and `b[l][m]==b[l][m+nNoDuplicate[l]-n[l]]`
#
# This required for example to model absolute value as two ReLU units with the negated weights and bias values.
struct Model
	n0::Int
	n::Array{Int,1}
	nNoDuplicate::Array{Int,1}
	L::Int
	W::Array{Array{Float64,2},1}
	b::Array{Array{Float64,1},1}
end
