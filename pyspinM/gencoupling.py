 def
 
 hMax1 = 1. / np.sqrt(np.sum(inv(obj.basisvector).T**2, axis=1))

    # Calculate the distance of the [1 1 1] point from the origin
    hMax2 = np.abs(np.sum(obj.basisvector, axis=1))

    # Calculate the closest point to the origin
    hMax = np.min([hMax1, hMax2])

    # Gives the number of extra unit cells along all 3 axes
    nC = np.ceil(param['maxDistance'] / hMax).astype(int)