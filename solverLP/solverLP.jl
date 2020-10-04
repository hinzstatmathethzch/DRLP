include("util/helpers.jl")
include("util/debugtools.jl")

function solverLP(model,x0)
trace=[]
##############################
# findVertex
##############################
state=SolverState(model,x0)
println(f(model,state.x)[1])

while size(state.critical,1) < model.n0
	(pos,normalvec)=step(model,state)
	push!(trace,(deepcopy(state),deepcopy(normalvec), f(model,state.x)[1]))
	println(f(model,state.x)[1])
end

##############################
# change the region
##############################
index=1
while true
pos=state.critical[index]
newActivation=!state.s[pos[1]][pos[2]]
changeActivationPattern!(state.s,model,pos,newActivation)
# compute new axis
normalvec=orientedNormalVec(model,pos,state.s)
# normalvec=-normalvec
projected=projectSkip(model,state, normalvec,index)
orthogonal=normalvec-projected
axis=(1/dot(orthogonal,normalvec))* orthogonal
state.Apseudo[index,:]=axis
# compute new gradient
state.gradient = gradient(model,state.s)
##############################
# select best axis to continue
##############################
state.Apseudo=computeAPseudo(model,state)
cd=coordinateDiff(model,state)
if cd>0.001
	throw("Coordinate difference diverged: $cd")
elseif cd >0.00001
	@warn("Coordinate difference large: $cd")
end
push!(trace,(deepcopy(state),deepcopy(normalvec), f(model,state.x)[1]))
state=deepcopy(trace[end][1])
i=bestIndex(state,-state.gradient)
if i>0
	state.direction=state.Apseudo[i,:]
	state.Apseudo= PseudoInverseRemoveCol(state.Apseudo,i)
	deleteat!(state.critical,i)
	(pos,normalvec,t)=step(model,state)
	# if t>0 && f(model,state.x)[1]>last(last(trace))
	# 	return (-2,trace,state.x) #-2 = increases in function value
	# end
	if pos[1]<=0 
		#if pos[1]==0 then gradient is zero, 
		#if pos[1]==-1 then function can be made arbitrarily small
		return (pos[1],trace,state.x) #return information code "pos[1]"
	end
	index=1
	println(f(model,state.x)[1])
else
	index+=1
	if index>model.n0
		println("finished")
		break;
	end
end
end
return (1,trace,state.x) # 1="converged"
end
