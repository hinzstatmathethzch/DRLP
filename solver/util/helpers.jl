include("../../models/model.jl")
using LinearAlgebra

####################
# Define the neural network itself 
# and similar functions
####################
htilde(m::Model,layerNum,input)= m.W[layerNum]*input+m.b[layerNum]
ReLU(x)=max(0,x)
h(m::Model,l,x)=ReLU.(htilde(m,l,x))
function f(m::Model,x)
	tmp=x
	for l = 1:m.L
		tmp=h(m,l,tmp)
	end
	return htilde(m,m.L+1,tmp)
end

####################
# function to compute the signature 
# and the gradient at a given point x
####################
function sigGrad(m::Model,x)
	s=Array{Array{Bool,1},1}(undef,m.L)
	grad=Matrix{Float64}(I,m.n0,m.n0)
	tmp=x
	for l = 1:m.L
		tmp=htilde(m,l,tmp)
		s[l]=Array{Bool,1}(undef,m.n[l])
		for j = 1:m.nNoDuplicate[l]
			s[l][j]=(tmp[j]>0)
		end
		diff=m.n[l]-m.nNoDuplicate[l]
		for j = (m.nNoDuplicate[l]+1):m.n[l]
			s[l][j]=!s[l][j-diff]
		end
		# s[l]=(tmp.>0)
		grad=s[l].*m.W[l]*grad
		tmp=ReLU.(tmp)
	end
	grad=m.W[m.L+1]*grad
	return (s,grad[1,:])
end

function sortfunction(x,y)
	if x[1]<y[1]
		return true;
	elseif x[1]==y[1]
		if x[2]<y[2]
			return true;
		end
	end
	return false
end

function gradient(m::Model,s)
	γ=m.W[m.L+1]
	for l = (m.L):-1:1
		γ=γ.*s[l]'*m.W[l]
	end
	return γ[1,:]
end
function changeActivationPattern!(s,m::Model,pos::Tuple{Int,Int},newVal::Bool)
	l=pos[1]
	j=pos[2]
	s[l][j]=newVal
	diff=m.n[l]-m.nNoDuplicate[l] #TODO: needs correction!! see DRLPL1
	if diff>0
		lower=m.nNoDuplicate[l]-diff+1
		upper=m.nNoDuplicate[l]
		if j>=lower || j<=upper
			s[l][j+diff]=!newVal
		elseif j>upper
			s[l][j-diff]=!newVal
		end
	end
end
function PseudoInverseRemoveCol(Apseudo,colId::Int)
	axis=Apseudo[colId,:]
	AnewPseudo=zeros(size(Apseudo,1)-1,size(Apseudo,2))
	i2=1;
	for i1 = 1:(size(Apseudo,1)-1)
		if i2==colId
			i2+=1
		end
		currentAxis=Apseudo[i2,:]
		newAxis=currentAxis-dot(currentAxis,axis)/ dot(axis,axis)*axis
		AnewPseudo[i1,:]=newAxis
		i2+=1
	end
	return AnewPseudo
end
mutable struct SolverState
	Apseudo::Array{Float64,2}
	s::Array{Array{Bool,1},1}
	x::Array{Float64,1}
	critical::Array{Tuple{Int,Int},1}
	gradient::Array{Float64,1}
	direction::Array{Float64,1}
	function SolverState(model::Model,x::Array{Float64,1})
		(s,grad)=sigGrad(model,x);
		new(zeros(0,model.n0),s,x,Array{Tuple{Int,Int},1}(),grad,-grad)
	end
end
function step(model::Model,state::SolverState)
	v=state.direction
	if norm(v)<1e-8
		throw("Direction is zero")
		# return (-1, nothing) #code -1: direction v is zero
	end
	(t,pos)=advanceMax(model,state,v)
	if t==Inf64 #no step possible
		throw("No step possible (infinite region)")
	end
	state.x=state.x+t*v
	############################### add critical
	normalvec=addCritical!(model,state,pos)
	if size(state.Apseudo,1)<model.n0
		############################### orthogonalize
		v=state.direction
		state.direction=v-project(model,state,v)
	else
		state.direction=zeros(model.n0)
	end
	return (pos,normalvec,t)
end
function bestIndex(state::SolverState,gradient::Vector)
maxVal=0.0;
maxIndex=0;
for i = 1:size(state.Apseudo,1)
	newVal=dot(state.Apseudo[i,:],gradient)/sqrt(dot(state.Apseudo[i,:],state.Apseudo[i,:]))
	if newVal>maxVal
		maxIndex=i;
		maxVal=newVal
	end
end
return maxIndex
end
function addCritical!(model::Model,state::SolverState,pos::Tuple{Int,Int})
	# println("pos: $(pos), state.s: $(state.s)")
	normalvec=orientedNormalVec(model,pos,state.s)
	# println("Normalvec: $(normalvec)")
	iproducts=innerProductsOrientedNormalVectors(model,state.s,normalvec)
	criticalInnerProducts = extractCriticalInnerProducts(state.critical,iproducts)
	state.Apseudo=PseudoInverseAddColIP(state.Apseudo,criticalInnerProducts,normalvec)
	push!(state.critical,pos)
	return normalvec
end
function ReLUArguments(m::Model,state::SolverState,x::Array{Float64,1})
	s=state.s
	out=Array{Array{Float64,1},1}()
	α=m.W[1]*x+m.b[1]
	for i = 1:m.L
		push!(out,α)
		α= m.W[i+1]*(s[i].*α)+m.b[i+1]
	end
	return out
end

####################
# project a vector "vec" onto the span of the critical 
# axes without the axis having index "skipIndex"
####################
function projectSkip(model::Model,state::SolverState,vec,skipIndex::Int)
	iproductsNegGradient=innerProductsOrientedNormalVectors(model,state.s,vec)
	coordinates = extractCriticalInnerProducts(state.critical,iproductsNegGradient)
	coordinates[skipIndex]=0.0;
	return state.Apseudo'*coordinates
end

function project(model::Model,state::SolverState,vec)
	if size(state.Apseudo,1)==0
		return zeros(model.n0)
	end
	if size(state.Apseudo,1)<model.n0
		iproductsNegGradient = innerProductsOrientedNormalVectors(model,state.s,vec)
		coordinates = extractCriticalInnerProducts(state.critical,iproductsNegGradient)
		return state.Apseudo' * coordinates
	end
	return vec
end

####################
# the advancemax algorithm step at position x
# in direction v with signature s ignoring 
# signature changes in neurons specified in "critical"
####################
function advanceMax(m::Model,state::SolverState,v,cap::Float64=-0.1)
	s=state.s
	x=state.x
	critical=deepcopy(state.critical)
	sort!(critical,lt=sortfunction)
	criticalIdx=1;
	considerCritical=(criticalIdx<=length(critical))
	nextCriticalLayer=0;
	if considerCritical
		nextCriticalLayer=first(critical[criticalIdx])	
	end
	change=Tuple{Int,Int}((1,1))
	α=m.W[1]*x+m.b[1]
	β=m.W[1]*v
	t=Inf64
	for i = 1:m.L
		for j = 1:m.n[i]
			if considerCritical
				if nextCriticalLayer==i
					if critical[criticalIdx][2]==j
						criticalIdx+=1
						if criticalIdx>length(critical)
							considerCritical=false
						else
							nextCriticalLayer=first(critical[criticalIdx])	
						end
						continue;
					end
				end
			end
			if j>m.nNoDuplicate[i]
				continue;
			end
			if β[j]!=0
				τ=-α[j]/β[j]
				if (((s[i][j]==true)&&(β[j]<0))||((s[i][j]==false)&&(β[j]>0))) && τ>cap
					if τ<t
						change=(i,j)
						t=τ
						if τ<t-1e-10
							#warning: maybe multiple hyperplanes here!
						end
					end
				end
			end
		end
		α= m.W[i+1]*(s[i].*α)+m.b[i+1]
		β= m.W[i+1]*(s[i].*β)
	end
	return (t,change)
end
function innerProductsOrientedNormalVectors(m::Model,s,v::Array{Float64,1})
	out=Array{Array{Float64,1},1}()
	γ=(m.W[1])*v
	for l = 1:m.L
		r=Array{Float64,1}(undef,m.n[l])
		sl=s[l]
		for j = 1:m.n[l]
			if sl[j]==true
				r[j]=γ[j]
			else
				r[j]=-γ[j]
			end
		end
		push!(out,r)
		γ= m.W[l+1]*(s[l].*γ)
	end
	return out
end
function orientedNormalVec(m::Model,neuronPos,s)
	γ=m.W[neuronPos[1]][neuronPos[2],:]
	for l = (neuronPos[1]-1):-1:1
		γ=m.W[l]'*(γ.*s[l])
	end
	if s[neuronPos[1]][neuronPos[2]]==false
		return -γ
	else 
		return γ
	end
end
function PseudoInverseAddColIP(Apseudo,AinnerProducts::Array{Float64,1},newCol::Array{Float64,1})
	# construct AnewPseudo
	AnewPseudo=zeros(size(Apseudo)[1]+1,size(Apseudo)[2])
	orthogonalPart=newCol-Apseudo'*AinnerProducts
	for i = 1:size(Apseudo,1)
		axis=Apseudo[i,:]
		adjustedAxis=axis-dot(axis,newCol)/dot(orthogonalPart,newCol)*orthogonalPart
		AnewPseudo[i,:]=adjustedAxis
	end
	AnewPseudo[size(AnewPseudo,1),:]=1/(dot(orthogonalPart,newCol))*orthogonalPart
	return AnewPseudo
end
function extractCriticalInnerProducts(criticalPositions::Array{Tuple{Int,Int},1},innerProducts)
	out=Array{Float64,1}(undef,size(criticalPositions,1))
	for i = 1:length(criticalPositions)
		out[i]=innerProducts[criticalPositions[i][1]][criticalPositions[i][2]]
	end
	return out
end
