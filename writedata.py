#!/usr/bin/python
from readsnapsgl import readsnapsgl
import numpy as np
from struct import *
import os
import time

def writedata(wpath,rpath,exts,spnm,binsize,fnum,ii,lgid,edn=None):
    """
    Writing data into mesh bins
    wpath : where mesh files are written
    rpath : where the simulation snapshot located
    exts: snapshot number
    spnm: snapshot name.
    edn   :  endian
    """

    print "Meshing snap_shot",ii

    if edn==None:
        edn='='

    np.savetxt(wpath+"indicator"+"."+str(ii), [1])

    if fnum<=1:
        head=readsnapsgl(rpath+"/"+spnm+exts,"HEAD",endian=edn)
    else:
        head=readsnapsgl(rpath+"/"+spnm+exts+"."+str(ii),"HEAD",endian=edn)
    npart=head[0]
    cumpart=np.zeros(npart.size+1,dtype=np.intp)
    for i in np.arange(npart.size):
        cumpart[i+1]=cumpart[i]+npart[i]
    boxsize=head[5]
    bins=np.intp(np.ceil(boxsize/binsize))
    bins2=bins*bins
    bins3=bins2*bins

    if fnum<=1:
        ID=readsnapsgl(rpath+"/"+spnm+exts,"ID  ",endian=edn,quiet=1,longid=lgid)
        pos=readsnapsgl(rpath+"/"+spnm+exts,"POS ",endian=edn,quiet=1)
    else:
        ID=readsnapsgl(rpath+"/"+spnm+exts+"."+str(ii),"ID  ",endian=edn,quiet=1,longid=lgid)
        pos=readsnapsgl(rpath+"/"+spnm+exts+"."+str(ii),"POS ",endian=edn,quiet=1)
    xyz=np.uint32(pos/binsize)
    xyz=xyz[:,0]*bins2+xyz[:,1]*bins+xyz[:,2]
    #H=np.zeros(bins3,dtype='uint32')
    mpart=np.zeros((bins3,npart.size),dtype=np.uint32)
    for i in range(npart.size):
        for j in range(cumpart[i],cumpart[i+1]):
            mpart[xyz[j],i]+=1

    H=np.sum(mpart,axis=1,dtype=np.int64)
    hcum=np.cumsum(H,dtype=np.int64)
    xyz=np.argsort(xyz)
    if fnum<=1:
        #pot=readsnapsgl(rpath+"/"+spnm,"POT ",endian=edn,quiet=1)
        mass=readsnapsgl(rpath+"/"+spnm,"MASS",endian=edn,quiet=1)
    else:
        #pot=readsnapsgl(rpath+"/"+spnm+exts+"."+str(ii),"POT ",endian=edn,quiet=1)
        mass=readsnapsgl(rpath+"/"+spnm+exts+"."+str(ii),"MASS",endian=edn,quiet=1)

    for j in range(0,ii):
        while os.path.isfile(wpath+"indicator"+"."+str(j)):
            time.sleep(5)

    for i in np.arange(bins3):
        ff=open(wpath+"tdata_"+str(i)+".dat",'ab')

        ff.write(np.int32(H[i]))
        if len(mass) ==1:
            ff.write(np.float32(mass))
        else:
            ff.write(np.float32(0.0))
        ff.write(np.int32(npart.size))
        if H[i]>0:
            ff.write(mpart[i,:])
            tmpos=np.sort(xyz[hcum[i]-H[i]:hcum[i]])
            ff.write(ID[tmpos])
            ff.write(pos[tmpos,:])
            if len(mass) !=1:
                ff.write(mass[tmpos])
            #ff.write(pot[tmpos])
        ff.close()

    os.remove(wpath+"indicator"+"."+str(ii))
    return len(mass)

def readdata_smp(ii,tmpph,boxsize,binsize,bufsize,fnum,lgid):
    bins=np.intp(np.ceil(boxsize/binsize))
    bins2=bins*bins
    bins3=bins2*bins

    if lgid:
        meshids=np.array([],dtype='uint64')
    else:
        meshids=np.array([],dtype='uint32')
    meshpos=np.array([[],[],[]],dtype='float32')
    meshpos=meshpos.transpose()
    meshpot=np.array([],dtype='float32')
    meshmas=np.array([],dtype='float32')
    meshnum=np.zeros(27,dtype=np.intp)
    meshnpt=np.zeros((27,fnum,6),dtype=np.intp)
    mnm    =0
    xyz=np.intp(np.array([ii/bins2,np.mod(ii,bins2)/bins,np.mod(np.mod(ii,bins2),bins)]))

    for iii in [xyz[0]-1,xyz[0],xyz[0]+1]:
        if iii == -1:
            xx=bins-1
            warp_x=-boxsize
        elif iii == bins:
            xx=0
            warp_x=boxsize
        else:
            xx=iii
            warp_x=0
        for jjj in [xyz[1]-1,xyz[1],xyz[1]+1]:
            if jjj == -1:
                yy=bins-1
                warp_y=-boxsize
            elif jjj == bins:
                yy=0
                warp_y=boxsize
            else:
                yy=jjj
                warp_y=0
            for kkk in [xyz[2]-1,xyz[2],xyz[2]+1]:
                if kkk == -1:
                    zz=bins-1
                    warp_z=-boxsize
                elif kkk == bins:
                    zz=0
                    warp_z=boxsize
                else:
                    zz=kkk
                    warp_z=0
                nxyz=xx*bins2+yy*bins+zz

                opf=open(tmpph+"tdata_"+str(nxyz)+".dat",'rb')
                for fn in np.arange(fnum):
                    H,sglmass,nm =unpack('I f i',opf.read(4*3))
                    if sglmass!=0:
                        meshmas=np.array([sglmass])
                    tmpbfn=0
                    if H>0:
                        mpart=np.ndarray(shape=(nm),dtype=np.uint32,buffer=opf.read(nm*4))
                        if lgid:
                            tmpmids=np.ndarray(shape=H,dtype='uint64',buffer=opf.read(8*H))
                        else:
                            tmpmids=np.ndarray(shape=H,dtype='uint32',buffer=opf.read(4*H))
                        tmpmpos=np.ndarray(shape=(H,3),dtype='float32',buffer=opf.read(4*3*H))+ \
                            np.array([[warp_x,warp_y,warp_z]])
                        if sglmass==0:
                            tmpmmas=np.ndarray(shape=H,dtype='float32',buffer=opf.read(4*H))
                        #tmpmpot=np.ndarray(shape=H,dtype='float32',buffer=opf.read(4*H))
                        if ((iii==xyz[0])&(jjj==xyz[1])&(kkk==xyz[2])) | (bufsize>=binsize):
                            tmpbfn=H
                            meshnpt[mnm,fn,:]=mpart
                        else:
                            idinbuf=(tmpmpos[:,0]>=xyz[0]*binsize-bufsize)&(tmpmpos[:,0]<(xyz[0]+1)*binsize+bufsize)& \
                                (tmpmpos[:,1]>=xyz[1]*binsize-bufsize)&(tmpmpos[:,1]<(xyz[1]+1)*binsize+bufsize)& \
                                (tmpmpos[:,2]>=xyz[2]*binsize-bufsize)&(tmpmpos[:,2]<(xyz[2]+1)*binsize+bufsize)
                            for kkkk in range(6):
                                meshnpt[mnm,fn,kkkk]=len(tmpmids[idinbuf[np.sum(mpart[0:kkkk],dtype=np.int64): \
                                                                             np.sum(mpart[0:kkkk],dtype=np.int64)+mpart[kkkk]]])
                            tmpmids=tmpmids[idinbuf]
                            tmpbfn=len(tmpmids)
                            tmpmpos=tmpmpos[idinbuf,:]
                            if sglmass==0:
                                tmpmmas=tmpmmas[idinbuf]
                            #tmpmpot=tmpmpot[idinbuf]

                        meshids=np.append(meshids,tmpmids)
                        meshpos=np.append(meshpos,tmpmpos,axis=0)
                        if sglmass==0:
                            meshmas=np.append(meshmas,tmpmmas)
                        #meshpot=np.append(meshpot,tmpmpot)
                    meshnum[mnm]+=tmpbfn

                opf.close()
                mnm+=1

    return meshids,meshpos,meshpot,meshmas,meshnum,meshnpt

# def readdata_acc(ii,tmpph,boxsize,binsize,bufsize,fnum,lgid):
#     bins=np.int64(np.ceil(boxsize/binsize))
#     bins2=bins*bins
#     bins3=bins2*bins

#     meshids=[]
#     meshpos=[]
#     meshpot=[]
#     meshmas=[]
#     meshnum=np.zeros(27,dtype=np.intp)
#     mnm    =0
#     xyz=np.int64(np.array([ii/bins2,np.mod(ii,bins2)/bins,np.mod(np.mod(ii,bins2),bins)]))
    
#     opf=open(tmpph+"tdata_"+str(ii)+".dat",'rb')
#     H,sglmas,nm =unpack('I f i',opf.read(4*3))
#     opf.close()
#     tmpart =np.zeros(nm,dtype=np.uintp)

#     for iii in [xyz[0]-1,xyz[0],xyz[0]+1]:
#         if iii == -1:
#             xx=bins-1
#             warp_x=-boxsize
#         elif iii == bins:
#             xx=0
#             warp_x=boxsize
#         else:
#             xx=iii
#             warp_x=0
#         for jjj in [xyz[1]-1,xyz[1],xyz[1]+1]:
#             if jjj == -1:
#                 yy=bins-1
#                 warp_y=-boxsize
#             elif jjj == bins:
#                 yy=0
#                 warp_y=boxsize
#             else:
#                 yy=jjj
#                 warp_y=0
#             for kkk in [xyz[2]-1,xyz[2],xyz[2]+1]:
#                 if kkk == -1:
#                     zz=bins-1
#                     warp_z=-boxsize
#                 elif kkk == bins:
#                     zz=0
#                     warp_z=boxsize
#                 else:
#                     zz=kkk
#                     warp_z=0
#                 nxyz=xx*bins2+yy*bins+zz

#                 opf=open(tmpph+"tdata_"+str(nxyz)+".dat",'rb')
#                 mpart=[0]
#                 for fn in np.arange(fnum):
#                     H,sglmass,nm =unpack('I f i',opf.read(4*3))
#                     if sglmass!=0:
#                         meshmas=np.array([sglmass])
                     
#                     tmpbfn=0
#                     if H>0:
#                         mpart=np.ndarray(shape=(nm),dtype=np.uint32,buffer=opf.read(nm*4))
#                         if lgid:
#                             #tmpmids=list(unpack('='+str(H)+'Q',opf.read(8*H)))
#                             tmpmids=np.ndarray(shape=H,dtype='uint64',buffer=opf.read(8*H))
#                         else:
#                             #tmpmids=list(unpack('='+str(H)+'L',opf.read(4*H)))
#                             tmpmids=np.ndarray(shape=H,dtype='uint32',buffer=opf.read(4*H))
#                         bufpos=opf.read(4*3*H)
#                         tmpmpos=np.ndarray(shape=(H,3),dtype='float32',buffer=bufpos)+ \
#                                 np.array([[warp_x,warp_y,warp_z]])
#                         if sglmass==0:
#                             #tmpmmas=list('='+str(H)+'f',opf.read(4*H))
#                             tmpmmas=np.ndarray(shape=H,dtype='float32',buffer=opf.read(4*H))
#                         #tmpmpot=np.ndarray(shape=H,dtype='float32',buffer=opf.read(4*H))

#                         if ((iii==xyz[0])&(jjj==xyz[1])&(kkk==xyz[2])) | (bufsize>=binsize):
#                             tmpbfn=H
#                         else:
#                             mpart.setflags(write=True)
#                             cumpart=np.cumsum(mpart)
#                             idinbuf=(tmpmpos[:,0]>=xyz[0]*binsize-bufsize)&(tmpmpos[:,0]<(xyz[0]+1)*binsize+bufsize)& \
#                                 (tmpmpos[:,1]>=xyz[1]*binsize-bufsize)&(tmpmpos[:,1]<(xyz[1]+1)*binsize+bufsize)& \
#                                 (tmpmpos[:,2]>=xyz[2]*binsize-bufsize)&(tmpmpos[:,2]<(xyz[2]+1)*binsize+bufsize)
#                             for kk in range(nm):
#                                 bp=cumpart[kk]-mpart[kk]
#                                 ep=cumpart[kk]
#                                 mpart[kk]=len(np.where(idinbuf[bp:ep]==True)[0])
#                             #tmpmids=list(tmpmids[i] for i in idinbuf)
#                             tmpmpos=tmpmpos[idinbuf,:].tolist()
#                             tmpmids=tmpmids[idinbuf].tolist()
#                             tmpbfn=len(tmpmids)
#                             if sglmass==0:
#                                 #tmpmmas=list(tmpmmas[i] for i in idinbuf)
#                                 tmpmmas=tmpmmas[idinbuf].tolist()
#                             #tmpmpot=tmpmpot[idinbuf].tolist()

#                         cumtpart=np.cumsum(tmpart)
#                         cumpart =np.cumsum(mpart)
#                         for kk in range(nm):
#                             if mpart[kk]>0:
#                                 bp=cumpart[kk]-mpart[kk]
#                                 ep=cumpart[kk]
#                                 loc_i=cumtpart[kk]+bp
#                                 meshids[loc_i:loc_i]=tmpmids[bp:ep]
#                                 meshpos[loc_i:loc_i]=tmpmpos[bp:ep]
#                                 meshmas[loc_i:loc_i]=tmpmmas[bp:ep]
#                                 #meshpot[loc_i:loc_i]=tmpmpot[bp:ep]
#                         tmpart+=mpart
#                     meshnum[mnm]+=tmpbfn

#                 opf.close()
#                 mnm+=1

#     if lgid:
#         meshids=np.array(meshids,dtype='uint64')
#     else:
#         meshids=np.array(meshids,dtype='uint32')
#     meshpos=np.array(meshpos,dtype='float32')
#     #meshpot=np.array(meshpot,dtype='float32')
#     meshmas=np.array(meshmas,dtype='float32')
#     return meshids,meshpos,meshpot,meshmas,tmpart,meshnum
