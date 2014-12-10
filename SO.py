#!/usr/bin/python
from readsnapsgl import readsnapsgl
import numpy as np
from time import time
from ckdtree import cKDTree
from analymesh import grouping, grouping_nl
from writedata import writedata,readdata_smp
#from dprofile import profile
from mpi4py import MPI
import os
import argparse
import ConfigParser

#init
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
if rank==0:
    print "Processes,", size
    st=time()

#agrparse
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,add_help=True)
parser.add_argument('pfile', nargs='?', default="param.txt", \
    help='input parameter file. If nothing is given, we try with param.txt')
parser.add_argument('-skip', type=int, default=0, dest='skip',\
    help='Skipping 1 (-skip 1) or 2 (-skip 2) part of the calculation')
parser.add_argument('--version', action='version', version='PIAO, 1.1')
results = parser.parse_args()
if rank==0:
    print "Tring to read parameter file : ", results.pfile

config = ConfigParser.RawConfigParser()
config.read(results.pfile)
wrtpth=config.get('ipp', 'wrtpth')
binsize=config.getfloat('ipp','binsize')
bufsize=config.getfloat('ipp','bufsize')
Numcut=config.getfloat('ipp','Numcut')
nbs=config.getint('ipp','nbs')
overlap=config.getboolean('ipp','overlap')
longid=config.getboolean('ipp','longid')
phot=config.get('ipp', 'phot')
phot=phot.strip('[]').split(',')
if rank==0:
    print "Overdensity was set to ", phot

snapth=config.get('sip','snapth')
snn=config.getint('sip','snn')
fnum=config.getint('sip','fnum')
spnm=config.get('sip','spnm')
edn=config.get('sip','edn')

if rank==0:
    if results.skip==0:
        print "We run the whole program, :)"
    else:    
        print "Try to skip the", results.skip, " part of the calculation" 

if snn == -1:
    exts=''
else:
    exts='000'+str(snn)
    exts=exts[-3:]

if fnum<=1:
    head=readsnapsgl(snapth+"/"+spnm+exts,"HEAD",endian=edn)
else:
    head=readsnapsgl(snapth+"/"+spnm+exts+".0","HEAD",endian=edn)

bmpc=str(np.intp(np.round(binsize/1000.)))+"Mpc"
tmpph =wrtpth+"tmpdata_"+exts+"_"+bmpc+"/"
boxsize=head[5]
if boxsize < 2000.:
    print "In Mpc/h? try to change the unit into Kpc/h, if not try to change me at here!"
    boxsize*=1000.
bins=np.intp(np.ceil(boxsize/binsize))
bins2=bins*bins
bins3=bins2*bins
if bins>=1000:
    print "We do not recommend to use such large number sub boxes : ", bins3

HubbleParam,Omega0,OmegaLambda=head[8],head[6],head[7]
redshift,scfa = head[3],head[2]
#pho_crit = 2.7753619773421899e-08  #Unit [10^10 M_sun/h] / [(kpc/h)^3] at redshift 0
pho_crit = 2.7846602822750852e-08  #Unit [10^10 M_sun/h] / [(kpc/h)^3] at redshift 0
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
if results.skip != 1:  #SKIP step 1
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
                spnm,binsize,fnum,ii,longid,edn=edn)
            comm.send(rank,dest=0,tag=1)
            Final=comm.recv(source=0,tag=2)
if rank==0:
    iot=time()
    print "meshing time:",time()-st

##Step two, SO groups, analysis each mesh one by one 
if results.skip != 2:  #SKIP step 2
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
            meshids,meshpos,meshpot,meshmas,meshnum,meshnpt= \
                readdata_smp(ii,tmpph,boxsize,binsize,bufsize,fnum,longid)
            #meshids,meshpos,meshpot,meshmas,tmpart,meshnum=readdata_acc(
            #    ii,tmpph,boxsize,binsize,bufsize,fnum,longid)
            Mtree=cKDTree(meshpos,leafsize=nbs/2)
            dens=Mtree.qdens(meshpos,meshmas,nbs,phoc=SOpho.min())
            #print "denstime",time()-st

            for ot in range(len(phot)):
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
