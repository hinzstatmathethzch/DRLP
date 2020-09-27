include("models/random.jl")
include("models/L1DataModel.jl")
include("./solverLP/solverLP.jl")


randmodel=randomModel(4,[5,4,2])
n=500
# input parameters
Ydata=rand(Uniform(-3,3),n)
Xdata=rand(Uniform(-3,3),randmodel.n0,n)

l=2
datamodel=L1DataModel(randmodel,Xdata,Ydata,l)
model=datamodel

x0=rand(Uniform(-20,20),model.n0)
x=x0

(code,trace1,x)=solverLP(model,x);
code

##############################
# analysis for increasing function value
##############################
#copy state and trace
statecopy=deepcopy(first(last(trace1)))
trace=deepcopy(trace1)
state=deepcopy(statecopy)

state.x
maximum(criticalCoordinates(model,state,state.critical))
differentSignatureCoordinates(model,statecopy)
setdiff(differentSignaturePositions(model,state),state.critical)
det(state.Apseudo)

# perform step
state=deepcopy(statecopy)
i=bestIndex(state,-state.gradient)
state.Apseudo=computeAPseudo(model,state)
state.negModifiedGrad=state.Apseudo[i,:]
state.Apseudo = PseudoInverseRemoveCol(state.Apseudo,i)
deleteat!(state.critical,i)

v=state.negModifiedGrad
(t,change)=advanceMaxAdjustedNew(model,state,v)

(pos,normalvec)=step(model,state)

(s1,g)=sigGrad(model,statecopy.x)
(s2,_)=sigGrad(model,state.x)
dot(g,statecopy.Apseudo[i,:])

s1.=statecopy.s
s2=state.s


differentSignatureCoordinates(model,statecopy)
differentSignatureCoordinates(model,state)

criticalCoordinates(model,statecopy,statecopy.critical)
criticalCoordinates(model,state,state.critical)

f(model,state.x)
f(model,statecopy.x)

l=3
sum(s1[l].==s2[l])


v=statecopy.Apseudo[i,:]
dot(v,statecopy.gradient)
(t,change)=advanceMaxAdjustedNew(model,state,v)
x2=statecopy.x+t*v
f(model,x2)

(s3,_)=sigGrad(model,statecopy.x+0*t*v)

s1[2][8]

sum(s3[2].==s1[2])

s3[2][8]
# END analysis for increasing function value
##############################


##############################
# analysis for negative t
##############################
(s,g)=sigGrad(model,x0)
#
trace=[]
state=SolverState(model,x0)
println(f(model,state.x)[1])
while size(state.critical,1) < model.n0
	v=state.negModifiedGrad
	(t,change)=advanceMaxAdjustedNew(model,state,v)
	println("t=$t")
	if t<0 
		println("t<0")
		break
	end
	(pos,normalvec)=step(model,state)
	s2=first(sigGrad(model,state.x))
	println("sum: ",sum(s2[2].==s[2]))
	if pos[1]<=0
		#if pos[1]==0 then gradient is zero, 
		#if pos[1]==-1 then function can be made arbitrarily small
		return (pos[1],trace,state.x)
	end
	push!(trace,(deepcopy(state),deepcopy(normalvec), f(model,state.x)[1]))
	println(f(model,state.x)[1])
end

state=deepcopy(first(trace[3]))
s2=first(sigGrad(model,state.x))
println("sum: ",sum(s2[2].==s[2]))
v=state.negModifiedGrad;

(t,change)=advanceMaxAdjustedNew(model,state,v);
println("t=$t")
# s2=first(sigGrad(model,state.x+t*v))
(pos,normalvec)=step(model,state)
s2=first(sigGrad(model,state.x))

state.critical
println("sum: ",sum(s2[2].==s[2]))

changes2=(s2[2].!=s[2])
pos=[(2,j) for j in findall(x->s2[2][x]!=s[2][x],1:model.n[2])]

pos=[(2,j) for j in 1:model.n[2]]
cc=criticalCoordinates(model,state,pos)
[(i,cc[i]) for i in 1:size(cc,1)]


m=model


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
change=Array{Tuple{Int,Int,Bool,Float64},1}()
sizehint!(change,10)
α=m.W[1]*x+m.b[1]
β=m.W[1]*v
t=Inf64
for i = 1:m.L
global considerCritical,nextCriticalLayer,criticalIdx,t,α,β
for j = 1:m.n[i]
if considerCritical
	if nextCriticalLayer==i
		if critical[criticalIdx][2]==j
			println("critical $i,$j")
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
if i==2 &&changes2[j]==true
	println(τ)
end
if ((s[i][j]==true)&&(β[j]<0))||((s[i][j]==false)&&(β[j]>0))
if τ<=t+1e-10
if τ<t-1e-10
empty!(change)
t=τ
end
push!(change,(i,j,(β[j]>0),τ))
end
end
end
end
α= m.W[i+1]*(s[i].*α)+m.b[i+1]
β= m.W[i+1]*(s[i].*β)
end

return (t,change)
end




