
######
# O(n0^3) pseudo matrix computation
function computeAPseudo(model,state)
	A=Matrix{Float64}(undef,model.n0,size(state.critical,1))
	for i = 1:size(state.critical,1)
		A[:,i]= orientedNormalVec(model,state.critical[i],state.s)
	end
	return inv(A'A)A'
end

function pseudoDiff(model::Model,state)
	maximum(abs.(state.Apseudo-computeAPseudo(model,state)))
end

######
# position correction
function criticalCoordinates(model,state,pos)
	args=ReLUArguments(model,state,state.x)
	return Array{Float64,1}([args[l][j] for (l,j) in pos])
end
function correctPosition(model,state)
	addCoordinates=criticalCoordinates(model,state,state.critical)
	state.x=state.x-state.Apseudo*addCoordinates
end
function coordinateDiff(model,state)
	return maximum(abs.(criticalCoordinates(model,state,state.critical)))
end


######
# different signature Coordinates (should be close to 0)
function differentSignaturePositions(model,state)
(s,_)=sigGrad(model,state.x)
indices=Array{Tuple{Int,Int},1}()
for i = 1:size(s,1)
for j = 1:size(s[i],1)
if s[i][j]!=state.s[i][j]
push!(indices,(i,j))
end
end
end
return indices
end
function differentSignatureCoordinates(model,state)
	pos=differentSignaturePositions(model,state)
	return criticalCoordinates(model,state,pos)
end
