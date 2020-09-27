## remove one col
function PseudoInverseRemoveCol(A,Apseudo,colId::Int)
	axis=Apseudo[colId,:]
	Anew=zeros(size(A)[1],size(A)[2]-1)
	AnewPseudo=zeros(size(A)[2]-1,size(A)[1])
	i2=1;
	for i1 = 1:(size(Apseudo,1)-1)
		if i2==colId
			i2+=1
		end
		Anew[:,i1]=A[:,i2]
		currentAxis=Apseudo[i2,:]
		newAxis=currentAxis-dot(currentAxis,axis)/ dot(axis,axis)*axis
		AnewPseudo[i1,:]=newAxis
		i2+=1
	end
	return (Anew,AnewPseudo)
end


## add one col
function PseudoInverseAddCol(A,Apeudo,newCol)
	# construct Anew
	Anew=zeros(size(A)[1],size(A)[2]+1)
	for j = 1:size(A,2)
		Anew[:,j]=A[:,j]
	end
	Anew[:,size(Anew,2)]=newCol
	# construct AnewPseudo
	AnewPseudo=zeros(size(A)[2]+1,size(A)[1],)
	orthogonalPart=newCol-Apseudo'*A'*newCol
	for i = 1:size(A,2)
		axis=Apseudo[i,:]
		adjustedAxis=axis-dot(axis,newCol)/dot(orthogonalPart,newCol)*orthogonalPart
		AnewPseudo[i,:]=adjustedAxis
	end
	AnewPseudo[size(AnewPseudo,1),:]=1/(dot(orthogonalPart,newCol))*orthogonalPart
	return (Anew,AnewPseudo)
end
