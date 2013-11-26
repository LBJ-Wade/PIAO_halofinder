#!/usr/bin/python
from readsnapsgl import readsnapsgl
import numpy as np
from time import time
from ckdtree import cKDTree
from analymesh import grouping, grouping_nl
from writedata import writedata,readdata_smp,readdata_acc
from dprofile import profile
from mpi4py import MPI
import os

#init
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
if rank==0:
    print "Processes,", size
    st=time()

#****** Input Parameter *************#
wrtpth="/home/wgcui/data/MUPPI/data/"  #Where your tmp data and SO group info will be saved.
binsize=6000.                     #meshbin size, lager than bufsize, in Kpc/h
bufsize=2000.                        #buffer  size, lager than the most massive halo mass radius *2
Numcut=50.                           #Min particle number within each SO group
nbs=aa                               #SPH neighbours
overlap=True                         #True or False, allow halo to overlap or not
longid =False #True                        #If particle id requires long long, or not
phot=np.array(["500","VIR"])  #Add here all the interested overdensities
skip1 =True  #False                          #skip first meshing step
skip2 =False #True                          #skip Second meshing step

#******* Snapshot input *************#
snapth="/home/wgcui/data/MUPPI/" #snapshot file path
snn=64                               #snapshot output number
fnum=4                              #snapshot file number
spnm="snap_le_"                     #snapshot base name
edn='='
exts='000'+str(snn)
exts=exts[-3:]
head=readsnapsgl(snapth+"/snapdir_"+exts+"/"+spnm+exts+".0","HEAD",endian=edn)

bmpc=str(np.intp(np.round(binsize/1000.)))+"Mpc"
tmpph =wrtpth+"tmpdata_"+exts+"_"+bmpc+"/"
boxsize=head[5]
bins=np.int64(np.ceil(boxsize/binsize))
bins2=bins*bins
bins3=bins2*bins
HubbleParam,Omega0,OmegaLambda=head[8],head[6],head[7]
redshift,scfa = head[3],head[2]
pho_crit = 2.7753619773421899e-08  #Unit [10^10 M_sun/h] / [(kpc/h)^3] at redshift 0
phomean=pho_crit*Omega0
pho_crit = pho_crit*(Omega0*(1+redshift)**3+OmegaLambda) 
#pho_crit = phomean*(Omega0*(1+redshift)**3+OmegaLambda)   # Now it's mean 200!!
#change to critical density of redshift z
omegaz=Omega0*(1+redshift)**3/(Omega0*(1+redshift)**3+OmegaLambda)
phovir=(18.*np.pi**2+82.*(omegaz-1)-39.*(omegaz-1)**2)*pho_crit
SOpho=np.zeros((0),dtype=np.float64)
for ot in phot:
    if ot == "VIR":
        SOpho=np.append(SOpho,phovir)
    else:
        SOpho=np.append(SOpho,pho_crit*np.intp(ot))

if rank == 0:
    print "cosmology parameter:",HubbleParam,Omega0,OmegaLambda,"redshift",redshift,"Boxsize", boxsize
    print "output SO overdensity", SOpho, " for ", phot

#Step 1 meshing data
if (not skip1):  #SKIP step 1
    if rank ==0:
        if os.path.isdir(tmpph):
            os.system('rm -rf %s' %tmpph)
        os.mkdir(tmpph)

        ii=0
        while (ii<fnum):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            comm.send(ii,dest=b,tag=3)
            ii+=1
        for j in range (1,size):
            b=comm.recv(source=j,tag=1)
            comm.send(True,dest=b,tag=2)
    else:
        comm.send(rank,dest=0,tag=1)
        Final=comm.recv(source=0,tag=2)
        while not(Final):
            ii=comm.recv(source=0,tag=3)
            aa=writedata(tmpph,snapth,exts,
                spnm,binsize,ii,longid,edn=edn)
            comm.send(rank,dest=0,tag=1)
            Final=comm.recv(source=0,tag=2)
if rank==0:
    iot=time()
    print "meshing time:",time()-st

##Step two, SO groups, analysis each mesh one by one 
if (not skip2):  #SKIP step 2
    if rank ==0:
        for ot in phot:
            if overlap:
                outph =wrtpth+"Groups_"+ot+"_"+exts+"_"+bmpc+"_"+str(nbs)+"/"
            else:
                outph =wrtpth+"Groups_"+ot+"_"+exts+"_"+bmpc+"_"+str(nbs)+"_nl/"
            if not os.path.isdir(outph):
                #os.system('rm -rf %s' %outph)
                os.mkdir(outph)
                os.mkdir(wrtpth+"Groups_"+ot+"_"+exts+"_"+bmpc+"_"+str(nbs)+"_nl/")

        ii=0
        while (ii<bins3):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            comm.send(ii,dest=b,tag=3)
            ii+=1
        for j in range (1,size):
            b=comm.recv(source=j,tag=1)
            comm.send(True,dest=b,tag=2)
    else:
        comm.send(rank,dest=0,tag=1)
        Final=comm.recv(source=0,tag=2)
        while not(Final):
            ii=comm.recv(source=0,tag=3)
            print "Analyzing meshbin %(1)d on rank %(2)d .... \n" % {
                "1":ii,"2":rank}

            st=time()
            ret_mesh=readdata_smp(ii,tmpph,boxsize,binsize,bufsize,fnum,longid)
            meshids,meshpos,meshpot,meshmas,meshnum=ret_mesh
            #meshids,meshpos,meshpot,meshmas,tmpart,meshnum=readdata_acc(
            #    ii,tmpph,boxsize,binsize,bufsize,fnum,longid)
            Mtree=cKDTree(meshpos,leafsize=nbs/2)
            dens=Mtree.qdens(meshpos,meshmas,nbs,phoc=SOpho.min())
            #print "denstime",time()-st

            for ot in range(phot.size):
                phobase=phomean*SOpho[ot]/pho_crit*Omega0
                if overlap:
                    outph =wrtpth+"Groups_"+phot[ot]+"_"+exts+"_"+bmpc+"_"+str(nbs)+"/"
                    outfn ='SO'
                    outfiles=outph+outfn
                    ret_grop=grouping(outfiles,Numcut,boxsize,binsize,
                             bufsize,scfa,SOpho[ot],ii,meshids,meshpos,meshmas,dens,phobase)
                    #if phot[ot]=='VIR':
                    #        profile(ii,boxsize,binsize,bufsize,longid,outph,scfa,ret_mesh,ret_grop)
                #else:
                    outph =wrtpth+"Groups_"+phot[ot]+"_"+exts+"_"+bmpc+"_"+str(nbs)+"_nl/"
                    outfn ='SO'
                    outfiles=outph+outfn
                    grouping_nl(outfiles,Numcut,boxsize,binsize,
                             bufsize,scfa,SOpho[ot],ii,meshids,meshpos,meshmas,dens,phobase)
            comm.send(rank,dest=0,tag=1)
            Final=comm.recv(source=0,tag=2)

if rank==0:
    ft=time()
    print "Tot time:", (time()-st)/3600. ,"Hours"
    ff=open(wrtpth+"/info",'a')
    ff.write(outph+"\n")
    ff.write("Time: IO  "+str(iot-st)+"\t analyzing  "+str(ft-iot)+
             "\t with CPUs:"+str(size)+"\n")
    ff.close()

    #if os.path.isdir(tmpph):
    #    os.system('rm -rf %s' %tmpph)
