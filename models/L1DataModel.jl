
function randomInput(model::Model,n::Int)
	return rand(model.n0,n)
end
function layerOutput(m::Model,x::Array{Float64,1},l::Int)
	tmp=x
	for i = 1:l
		tmp=h(m,i,tmp)
	end
	return tmp
end
function layerOutput(m::Model,Xdata::Matrix{Float64},l::Int)
	n=size(Xdata,2)
	Xtransformed=Matrix{Float64}(undef,(l==0 ? m.n0 : m.n[l]),size(Xdata,2))
	for i = 1:n
		Xtransformed[:,i]=layerOutput(m,Xdata[:,i],l)
	end
	return Xtransformed
end
function diagonalMatrices(M::Array{Float64,2},times::Int)
	out=zeros(Float64,size(M,1)*times,size(M,2)*times)
	I=size(M,1)
	J=size(M,2)
	for k = 1:times
	for i = 1:I
	for j = 1:J
		out[(k-1)*I+i,(k-1)*J+j]=M[i,j]
	end
	end
	end
	return out
end
function fillXWeightmatrix!(W,ni0::Int,ni1::Int, ioffset::Int,x::Array{Float64,1})
for k = 1:ni1
	i=ioffset+k
	joffset=(k-1)*(ni0+1)
	for l = 1:ni0
		j=joffset+l
		W[i,j]=x[l]
	end
	W[i,joffset+ni0+1]=1
end
end
function constructXWeightmatrix(X::Array{Float64,2},ni1::Int)
	ni0=size(X,1)
	W=zeros(Float64,size(X,2)*ni1,ni1*(ni0+1))
	for i = 1:(size(X,2))
		ioffset=(i-1)*(ni1)
		fillXWeightmatrix!(W,ni0,ni1,ioffset,X[:,i])
	end
	return W
end
function L1DataModel(m::Model,X::Array{Float64,2},Y::Array{Float64,1},l::Int)
@assert size(Xdata,2)==size(Ydata,1)
@assert l<=m.L+1
#
##################### L
L=m.L+2-l
#
##################### W
Xtransformed=layerOutput(m,Xdata,l-1)
W1=constructXWeightmatrix(Xtransformed,(l<=m.L ? m.n[l] : 1))
W=Array{Array{Float64,2},1}()
b=Array{Array{Float64,1},1}()
if l<m.L+1
	Wlast=[diagonalMatrices(m.W[m.L+1],n);-diagonalMatrices(m.W[m.L+1],n)]
	W=Array{Array{Float64,2},1}([W1,[diagonalMatrices(m.W[i],n) for i in (l+1):(m.L)]...,Wlast, repeat([1],outer=(1,2*n))])
	b0=zeros(Float64,size(W1,1))
	blast= [repeat(m.b[m.L+1], outer=n)-Ydata;-repeat(m.b[m.L+1], outer=n)+Ydata]
	b=[b0,[repeat(m.b[i], outer=n) for i in (l+1):m.L]...,blast,[0]]
else #l== m.L+1
	b0=zeros(Float64,size(W1,1))
	blast=[-Ydata;Ydata]
	W=Array{Array{Float64,2},1}([[W1; -W1], repeat([1],outer=(1,2*n))])
	b=[blast,[0]]
end
#
##################### b
#
##################### widths
n0=size(W1,2)
widths=[[n.*m.n[i] for i in (l):m.L]...,2*n]
noduplicate=[[n.*m.n[i] for i in (l):m.L]...,n]
return Model(n0,widths,noduplicate,L,W,b)
end
