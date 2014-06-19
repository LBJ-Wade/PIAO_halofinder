PIAO
====

PIAO (漂) is efficient memory-controlled python code using the standard SO algorithm to identify halos.


Reqiures (not fully checked!): 
    Python >= 2.6.x, not know for 3.x.x
    Numpy  >= 1.5.x
    Cython >= 0.18
    Mpi4Py >= 1.2
    Scipy  >= 0.8.x (optimal)

To use:
    1, build ckdtree lib with Cython by simply running build_lib : ./build_lib
    2. Modify parameters in SO.py
    ##3. run the code by mpiexec -np 6 python SO.py
    3. Now, I include the parameter file, change the parameters in this param.txt file or write your own.
	But, it need to be kept its format (similar to windows INI format, see python ConfigParser for more	   detail). 
    4. Use ./SO.py -h for more

Note:
The ckdtree.pyx file is shamelessly stolen from https://github.com/scipy/scipy/blob/master/scipy/spatial/ckdtree.pyx.
Thanks to Patrick Varilly.
