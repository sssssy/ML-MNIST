import struct

def test_images() ->list:
    
    '''
        data = [ [0,0,0,0, ... ,37, 233,255,255, ... ,72,14,0,0], <-784项
                 [0,0,0,0, ... ,26, 230,255,255, ... ,60,11,0,0],
                 ...
               ] # / 255
    '''
    
    filename = 'test-images.idx3-ubyte'
    binfile = open(filename, 'rb')
    buf = binfile.read()
    binfile.close()

    index = 0
    magicnumber, numimages, numrows, numcols = struct.unpack_from('>iiii', buf, index)
    index += struct.calcsize('>iiii')
    
    n = numimages
    data = [[] for x in range(n)]
    for i in range(0,n):
        img = struct.unpack_from('784B', buf, index)
        data[i] = list(img)
        index += struct.calcsize('784B')
##    for i in range(len(data)):
##        for j in range(784):
##            data[i][j] = data[i][j] / 255

    return data
