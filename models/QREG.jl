function QREG_Model(X,Y,alpha::Float64)
	if alpha<0.0 || alpha>1.0
		error("Argument alpha=$(alpha) must be in [0,1].")	
	end
	N=size(X)[1]
	p=size(X)[2]
	Wx=[X repeat([1.0],outer=[N])]
	by=-Y
	#
	Wx2=[Wx;-Wx]
	by2=[by;-by]
	#
	W2=[repeat([1.0-alpha],outer=[1,N]) repeat([alpha],outer=[1,N])]
	W1=Wx2
	b1=by2
	b2=[0.0]
	#
	W=[W1,W2]
	b=[b1,b2]
	n0=p+1
	L=1
	n=[2N]
	nNoDuplicate=[N]
	return Model(n0,n,nNoDuplicate,L,W,b)
end
