PIAO
====

PIAO (漂) is efficient memory-controlled python code using the standard SO algorithm to identify halos.


Reqiures: 
    Python >= 2.6.x, not know for 3.x.x
    Numpy  >= 1.5.x
    Cython >= 0.18
    Mpi4Py >= 1.2
    Scipy  >= 0.8.x (optimal)

To use:
    1, build ckdtree lib with Cython by simply running build_lib : ./build_lib
    2. Modify parameters in SO.py
    3. run the code by mpiexec -np 6 python SO.py
    
Note:
The ckdtree.pyx file is shamelessly stolen from https://github.com/scipy/scipy/blob/master/scipy/spatial/ckdtree.pyx.
Thanks to Patrick Varilly.
