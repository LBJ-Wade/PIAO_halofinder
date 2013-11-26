import numpy as np
def match2arr(array1, array2, position, order=None):
        """
        Be ware of the position's dtype!
        It should match the array2's length.
        """

	ea1=array1.size
	ea2=array2.size
	nep=position.size

	if ea1 != nep:
		print "position array for saving is not match first array!"
		return(0)

	if order == 1: 
		sa1=np.argsort(array1)
		a1 =array1[sa1]
		sa2=np.argsort(array2)
		a2 =array2[sa2]
	else:
		a1=array1
		a2=array2

	matchnum=0
	i=0
        j=0

	while (i < ea1 and j < ea2):
		if (a1[i] == a2[j]):
			matchnum+=1
			if order == 1:
				position[sa1[i]] = sa2[j]
			else:
				position[i] = j
			i+=1
                        while (i < ea1-1 and a1[i] == a2[j]):
                                matchnum+=1
                                if order == 1:
                                        position[sa1[i]] = sa2[j]
                                else:
                                        position[i] = j
                                i+=1
			j+=1
		elif a1[i] > a2[j] :
			j+=1
		elif a1[i] < a2[j] :
			if order == 1:
				position[sa1[i]] = -1
			else : 
				position[i] = -1
			i+=1

	if i < ea1:
		if order == 1:
			position[sa1[i:]] = -1
		else:
			position[i:] = -1

	return(matchnum)
