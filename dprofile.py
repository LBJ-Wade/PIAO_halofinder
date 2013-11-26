#!/usr/bin/python
import numpy as np
from match2arr import  match2arr
import string

def profile(ii,boxsize,binsize,bufsize,lgid,wrtpth,scfa,meshinfo,goupinfo):

    #Read in data
    meshids,meshpos,meshpot,meshmas,meshnum = meshinfo
    #meshids,meshpos,meshpot,meshmas,tmpart,meshnum = readdata_acc(
    #    ii,tmpph,boxsize,binsize,bufsize,fnum,lgid)
    
    idn=string.find(wrtpth, "/")
    if idn ==-1:
        DMids=meshids[tmpart[0]:tmpart[0]+tmpart[1]]
        DMpos=meshpos[tmpart[0]:tmpart[0]+tmpart[1],:]
        DMmas=meshmas[tmpart[0]:tmpart[0]+tmpart[1]]
        stdmid=np.argsort(DMids)
        DMids=DMids[stdmid]
        DMpos=DMpos[stdmid,:]
        DMmas=DMmas[stdmid]

        stnbf=tmpart[:4].sum()
        STids=meshids[stnbf:stnbf+tmpart[4]]
        STpos=meshpos[stnbf:stnbf+tmpart[4],:]
        STmas=meshmas[stnbf:stnbf+tmpart[4]]
        ststid=np.argsort(STids)
        STids=STids[ststid]
        STpos=STpos[ststid,:]
        STmas=STmas[ststid]

    stmid=np.argsort(meshids)
    meshids=meshids[stmid]
    meshpos=meshpos[stmid,:]
    meshmas=meshmas[stmid]

    SONgroups,SONids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs=goupinfo
    #SONgroups,SONids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs=readgroups(wrtpth,outfl[7:],longid=lgid,sglnum=ii,readid=True)

    bnum=50
    binr=10.  #[Kpc/h]
    mtid=np.where(GroupMass>=10.**2.4)[0]
    #mtid=np.zeros(gcid.size,dtype="int32")-1
    #if match2arr(gcid,GroupPids,mtid,order=1) != gcid.size:
    #    print "Cannot match cid position!!",ii,gcid.size
    #    stop

    tdp=np.zeros((mtid.size,bnum+1),dtype='float32')-1.
    if idn ==-1:
        dmdp=np.zeros((mtid.size,bnum+1),dtype='float32')-1
        stdp=np.zeros((mtid.size,bnum+1),dtype='float32')-1

    for NN in range(mtid.size):
	gn=mtid[NN]
        rvir=GroupRad[gn]
        rbin=np.zeros(bnum+2,dtype='float32')
        lbs =-np.log10(binr/rvir)/bnum
        rbin[1:]=np.arange(np.log10(binr/rvir),lbs-1.0e-7,lbs)
        #aa=np.arange(np.log10(binr/rvir),-np.log10(binr/rvir)/bnum-0.00001,-np.log10(binr/rvir)/bnum)
        rbin[1:]=10.**rbin[1:]*rvir
        gids=GroupIDs[GroupOffset[gn]:GroupOffset[gn]+GroupLen[gn]]
        posgid=np.zeros(gids.size,dtype='int64')

        #For total dp
        stgid=np.argsort(gids)
        mm=match2arr(gids[stgid],meshids,posgid)
        if mm != gids.size:
            print "Error!",gn,mm,gids.size
            stop
        else:
            #gid0p=stgid[0]
            gtpos=meshpos[posgid,:]
            #cc=gtpos[gid0p,:]
            cc=GroupptPos[gn]
            rdius=np.sqrt(np.sum((gtpos-cc)**2,axis=1))
            gidmas=meshmas[posgid]
            bnmas,b=np.histogram(rdius,bins=rbin,weights=gidmas)
            tdp[NN,:]=np.cumsum(bnmas)/(4.*np.pi*(rbin[1:]*scfa)**3/3.)

        if idn ==-1:
            posidm=np.zeros(gids.size,dtype='int64')-1
            md=match2arr(gids[stgid],DMids,posidm)
            iddm=posidm>=0
            gdmpos=DMpos[posidm[iddm],:]
            rdsdm=np.sqrt(np.sum((gdmpos-cc)**2,axis=1))
            gdmmas=DMmas[posidm[iddm]]
            bdmas,b=np.histogram(rdsdm,bins=rbin,weights=gdmmas)
            dmdp[NN,:]=np.cumsum(bdmas)/(4.*np.pi*(rbin[1:]*scfa)**3/3.)

            posist=np.zeros(gids.size,dtype='int64')-1
            ms=match2arr(gids[stgid],STids,posist)
            idst=posist>=0
            gstpos=STpos[posist[idst],:]
            rdsst=np.sqrt(np.sum((gstpos-cc)**2,axis=1))
            gstmas=STmas[posist[idst]]
            bsmas,b=np.histogram(rdsst,bins=rbin,weights=gstmas)
            stdp[NN,:]=np.cumsum(bsmas)/(4.*np.pi*(rbin[1:]*scfa)**3/3.)

    ff=open(wrtpth+"/DP."+str(ii),'wb')
    ff.write(np.int32(mtid.size))
    ff.write(np.int32(bnum+1))
    ff.write(np.int32(mtid))
    ff.write(np.float32(GroupRad[mtid]))
    ff.write(np.float32(tdp))
    if idn ==-1:
        ff.write(np.float32(dmdp))
        ff.write(np.float32(stdp))
    ff.close()

    return 0
