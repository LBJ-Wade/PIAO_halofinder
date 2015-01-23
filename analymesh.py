import numpy as np
from time import time
from struct import *

def grouping(outfiles,Numcut,boxsize,binsize,bufsize,scfa,SOpho,ii,meshids,meshpos,meshmas,dens):

    bins=np.intp(np.ceil(boxsize/binsize))
    bins2=bins*bins
    bins3=bins2*bins
    if meshmas.size == 1:
        sglmass=True
    else:
        sglmass=False

    SOTotNgroups,SOTotNids=0,0
    SOGroupLen   = []  
    SOGroupOffset= [0]
    SOGroupMass  = np.array([],dtype=np.float32)
    SOR500       = []  
    SOpotPos     = []  
    SOmcPos      = []  
    SOPids       = []
    SOGroupIDs   =np.zeros(meshids.size,dtype=meshids.dtype)

    dens=np.abs(dens)
    nxyz=np.int64(np.array([ii/bins2,np.mod(ii,bins2)/bins,np.mod(np.mod(ii,bins2),bins)]))
    xyzmin,xyzmmin=nxyz*binsize,nxyz*binsize-bufsize
    xyzmax=(nxyz+1)*binsize
    lmbs=np.intp(binsize/50.)  #reseparate the meshbox into 125000 subboxes.
    lbn=np.int32(np.ceil((binsize+2*bufsize)/lmbs))
    xyz=np.uint32((meshpos-xyzmmin)/lmbs)
    xyz=xyz[:,0]*lbn*lbn+xyz[:,1]*lbn+xyz[:,2]
    H2=np.zeros(lbn**3,dtype='int32')
    for i in np.arange(xyz.size):
        H2[xyz[i]]+=1
    xyz=np.argsort(xyz)
    hcum2 =np.cumsum(H2,dtype=np.int64)
    Ncount=0

    while True:     #Loop for peaks and SO groups
        didp=dens.argmax()
        ppos=meshpos[didp,:]
        if (dens[didp]<SOpho) & (Ncount>100):
            break

        rrag=lmbs
        while True:
            lmg=np.int32(np.floor((ppos-xyzmmin-rrag)/lmbs))
            idlmg=lmg<0
            if len(lmg[idlmg])>0:
                lmg[idlmg]=0
            hmg=np.int32(np.ceil((ppos-xyzmmin+rrag)/lmbs))
            idhmg=hmg>lbn
            if len(hmg[idhmg])>0:
                hmg[idhmg]=lbn
            tmgrid=np.mgrid[lmg[0]:hmg[0],lmg[1]:hmg[1],lmg[2]:hmg[2]]
            tmgrid=np.reshape(tmgrid,(3,(hmg[0]-lmg[0])*(hmg[1]-lmg[1])*(hmg[2]-lmg[2]))).T
            tmgrid=tmgrid[:,0]*lbn*lbn+tmgrid[:,1]*lbn+tmgrid[:,2]
            lmgidp=np.array([],dtype='int32')
            for jj in tmgrid:
                if H2[jj]>0:
                    lmgidp=np.append(lmgidp,xyz[hcum2[jj]-H2[jj]:hcum2[jj]])
            
            Radius=np.sqrt(np.sum((meshpos[lmgidp]-ppos)**2,axis=1,dtype=np.float64))
            indrs =np.argsort(Radius)
            Radius=Radius[indrs]
            if sglmass:
                CRmas =np.arange(1,Radius.size+1,dtype='float64')*meshmas
            else:
                CRmas =meshmas[lmgidp[indrs]]
                CRmas =np.cumsum(CRmas,dtype=np.float64)
            CRad  =(Radius[1:]+Radius[:-1])/2.        
            Rho   =CRmas[:-1]/(4.*np.pi*(CRad*scfa)**3/3.)
            rindlr=np.where(Rho<=SOpho)[0]            
            if rindlr.size==0:
                rrag*=4.
            elif CRad[rindlr[0]] > 0.5*rrag: #keep the distance to bounder at least half sub-boxsize
                rrag*=3.
            else:
                rslpart=lmgidp[indrs[0:rindlr[0]+1]]
                gradius=CRad[rindlr[0]]
                break

        if rindlr[0]>=Numcut:
            rpxyz=np.int32(np.floor(ppos/binsize))
            if (rpxyz[0]==nxyz[0]) & (rpxyz[1]==nxyz[1]) &(rpxyz[2]==nxyz[2]):   #if the new postion out of base box, we will drop it 
                mtdenpos=np.argmax(np.abs(dens[rslpart]))
                if (dens[didp]<np.abs(dens[rslpart][mtdenpos])) & (len(SOpotPos)>0):  #other higher density peak in this group
                    distog=np.sqrt(np.sum((np.array(SOpotPos)-ppos)**2,axis=1,dtype=np.float64))  #check for already identified groups
                    mtid=np.where(distog<gradius)[0]
                    if (len(mtid)>0) & (meshids[rslpart][mtdenpos] in SOPids):  #Other identified groups lie in this group. Take them as its substructures and remove by M*-1
                        ## the second condition get rid of groups lying on the edge of big one.
                        SOGroupMass[mtid]=np.abs(SOGroupMass[mtid])*-1.
                        SOpotPos.append(ppos)            
                        SOGroupMass=np.append(SOGroupMass,CRmas[rindlr[0]])
                        SOR500.append(gradius)            
                        SOGroupLen.append(rindlr[0]+1)   
                        SOGroupOffset.append(rindlr[0]+1+SOGroupOffset[SOTotNgroups])
                        SOTotNids+=rindlr[0]+1
                        SOPids.append(meshids[didp])      
                        if sglmass:
                            SOmcPos.append(np.mean(meshpos[rslpart,:],axis=0,dtype=np.float64)) 
                        else:
                            SOmcPos.append(np.sum(meshpos[rslpart,:]*np.reshape(meshmas[rslpart],(rindlr[0]+1,1)),axis=0,dtype=np.float64)/SOGroupMass[-1])
                        SOGroupIDs[SOGroupOffset[SOTotNgroups]:SOTotNids]=meshids[rslpart]
                        SOTotNgroups+=1   
                    else:   ##We only taken out the particles(within half of the radius) of the density loop
                        rslpart=lmgidp[indrs[0:np.int32(rindlr[0]/2)]]
                else:
                    SOpotPos.append(ppos)                  
                    SOGroupMass=np.append(SOGroupMass,CRmas[rindlr[0]])
                    SOR500.append(gradius)             
                    SOGroupLen.append(rindlr[0]+1)      
                    SOGroupOffset.append(rindlr[0]+1+SOGroupOffset[SOTotNgroups]) 
                    SOTotNids+=rindlr[0]+1
                    SOPids.append(meshids[didp])    
                    if sglmass:
                        SOmcPos.append(np.mean(meshpos[rslpart,:],axis=0,dtype=np.float64))  
                    else:
                        SOmcPos.append(np.sum(meshpos[rslpart,:]*np.reshape(meshmas[rslpart],(rindlr[0]+1,1)),axis=0,dtype=np.float64)/SOGroupMass[-1])                  
                    SOGroupIDs[SOGroupOffset[SOTotNgroups]:SOTotNids]=meshids[rslpart]
                    SOTotNgroups+=1   
        else:
            Ncount+=1
        dens[rslpart]=-1*np.abs(dens[rslpart])

    SOGroupIDs=SOGroupIDs[:SOTotNids]
    ff=open(outfiles+"."+str(ii),'wb')
    d1=pack('q q q',SOTotNgroups,SOTotNids,bins3)
    ff.write(d1)
    ff.write(np.array(SOGroupLen,dtype=np.int32))
    ff.write(np.array(SOGroupOffset[:SOTotNgroups],dtype=np.uint32))
    ff.write(np.float32(SOGroupMass))
    ff.write(np.array(SOR500,dtype=np.float32))
    ff.write(np.array(SOpotPos,dtype=np.float32))
    ff.write(np.array(SOmcPos,dtype=np.float32))
    ff.write(np.array(SOPids))
    ff.write(SOGroupIDs)
    ff.close()

    return SOTotNgroups #,SOTotNids,SOGroupLen,SOGroupOffset[:SOTotNgroups],SOGroupMass,SOR500,SOpotPos,SOmcPos,SOPids,SOGroupIDs

def grouping_nl(outfiles,Numcut,boxsize,binsize,bufsize,scfa,SOpho,ii,meshids,meshpos,meshmas,dens):

    bins=np.intp(np.ceil(boxsize/binsize))
    bins2=bins*bins
    bins3=bins2*bins
    if meshmas.size == 1:
        sglmass=True
    else:
        sglmass=False

    SOTotNgroups,SOTotNids=0,0
    SOGroupLen   = []
    SOGroupOffset= [0] 
    SOGroupMass  = np.array([],dtype=np.float32)
    SOR500       = []  
    SOpotPos     = [] 
    SOmcPos      = [] 
    SOPids       = [] 
    SOGroupIDs   =np.zeros(meshids.size,dtype=meshids.dtype)
    
    dens=np.abs(dens)
    nxyz=np.int64(np.array([ii/bins2,np.mod(ii,bins2)/bins,np.mod(np.mod(ii,bins2),bins)]))
    xyzmin,xyzmmin=nxyz*binsize,nxyz*binsize-bufsize
    xyzmax=(nxyz+1)*binsize
    lmbs=np.intp(binsize/50.)
    lbn=np.int32(np.ceil((binsize+2*bufsize)/lmbs))
    xyz=np.uint32((meshpos-xyzmmin)/lmbs)
    xyz=xyz[:,0]*lbn*lbn+xyz[:,1]*lbn+xyz[:,2]
    H2=np.zeros(lbn**3,dtype='int32')
    for i in np.arange(xyz.size):
        H2[xyz[i]]+=1
    xyz=np.argsort(xyz)
    hcum2 =np.cumsum(H2,dtype=np.int64)
    blp=np.ones(meshids.size,dtype='bool')
    Ncount=0

    while True:     #Loop for peaks and SO groups
        didp=dens.argmax()
        ppos=meshpos[didp,:]
        if (dens[didp]<SOpho) & (Ncount>100):
            break

        rrag=lmbs
        while True:
            lmg=np.int32(np.floor((ppos-xyzmmin-rrag)/lmbs))
            idlmg=lmg<0
            if len(lmg[idlmg])>0:
                lmg[idlmg]=0
            hmg=np.int32(np.ceil((ppos-xyzmmin+rrag)/lmbs))
            idhmg=hmg>lbn
            if len(hmg[idhmg])>0:
                hmg[idhmg]=lbn
            tmgrid=np.mgrid[lmg[0]:hmg[0],lmg[1]:hmg[1],lmg[2]:hmg[2]]
            tmgrid=np.reshape(tmgrid,(3,(hmg[0]-lmg[0])*(hmg[1]-lmg[1])*(hmg[2]-lmg[2]))).T
            tmgrid=tmgrid[:,0]*lbn*lbn+tmgrid[:,1]*lbn+tmgrid[:,2]
            lmgidp=np.array([],dtype='int32')
            for jj in tmgrid:
                if H2[jj]>0:
                    lmgidp=np.append(lmgidp,xyz[hcum2[jj]-H2[jj]:hcum2[jj]])
            blgidp=lmgidp[blp[lmgidp]]
            Radius=np.sqrt(np.sum((meshpos[blgidp]-ppos)**2,axis=1,dtype=np.float64))
            indrs =np.argsort(Radius)
            Radius=Radius[indrs]
            if sglmass:
                CRmas =np.arange(1,Radius.size+1,dtype='float32')*meshmas
            else:
                CRmas =meshmas[blgidp[indrs]]
                CRmas =np.cumsum(CRmas,dtype=np.float64)
            CRad  =(Radius[1:]+Radius[:-1])/2.        
            Rho   =CRmas[:-1]/(4.*np.pi*(CRad*scfa)**3/3.)
            rindlr=np.where(Rho<=SOpho)[0]            
            if rindlr.size==0:
                rrag*=4.
            elif CRad[rindlr[0]]>0.5*rrag:
                rrag*=3.
            else:
                rslpart=blgidp[indrs[0:rindlr[0]+1]]
                gradius=CRad[rindlr[0]]
                break

        if rindlr[0]>=Numcut:
            rpxyz=np.int32(np.floor(ppos/binsize))
            if (rpxyz[0]==nxyz[0]) & (rpxyz[1]==nxyz[1]) & (rpxyz[2]==nxyz[2]):   #if the new postion out of base box, we will drop it           
                SOGroupMass=np.append(SOGroupMass,CRmas[rindlr[0]])
                SOPids.append(meshids[didp])   
                SOR500.append(gradius)      
                SOGroupLen.append(rindlr[0]+1)  
                SOGroupOffset.append(rindlr[0]+1+SOGroupOffset[SOTotNgroups]) 
                SOTotNids+=rindlr[0]+1
                SOGroupIDs[SOGroupOffset[SOTotNgroups]:SOTotNids]=meshids[rslpart]
                if sglmass:
                    SOmcPos.append(np.mean(meshpos[rslpart,:],axis=0,dtype=np.float64))
                else:
                    SOmcPos.append(np.sum(meshpos[rslpart,:]*np.reshape(meshmas[rslpart],(rindlr[0]+1,1)),axis=0,dtype=np.float64)/SOGroupMass[-1])
                if SOTotNgroups != 0: #check for other groups in this new identified group
                    distog=np.sqrt(np.sum((SOpotPos-ppos)**2,axis=1,dtype=np.float64))
                    idindt=np.where(distog<gradius)[0]
                else:
                    idindt=np.array([])
                SOpotPos.append(ppos)           #After checking
                if len(idindt)>0:     #True
                    SOTotNids-=rindlr[0]+1
                    Rho=(CRmas[:-1]+np.sum(SOGroupMass[idindt],dtype=np.float64))/(4.*np.pi*(CRad*scfa)**3/3.)
                    rindlr=np.where(Rho<=SOpho)[0]
                    llpart=len(rslpart)
                    rslpart=blgidp[indrs[0:rindlr[0]+1]]
                    SOGroupMass[-1]=CRmas[rindlr[0]]+np.sum(SOGroupMass[idindt],dtype=np.float64)
                    SOR500[-1]=CRad[rindlr[0]]
                    SOGroupMass[idindt]=-1*np.abs(SOGroupMass[idindt])
                    SOGroupLen[-1]=rindlr[0]+1
                    SOGroupOffset[-1]=rindlr[0]+1+SOGroupOffset[SOTotNgroups]
                    SOTotNids+=rindlr[0]+1
                    SOGroupIDs[SOGroupOffset[SOTotNgroups]:SOTotNids]=meshids[rslpart]
                    for rwnum in idindt:
                        SOGroupIDs[SOTotNids:SOTotNids+SOGroupLen[rwnum]]=SOGroupIDs[
                            SOGroupOffset[rwnum]:SOGroupOffset[rwnum]+SOGroupLen[rwnum]]
                        SOmcPos[-1]=(SOmcPos[-1]*CRmas[rindlr[0]]-SOmcPos[rwnum]*SOGroupMass[rwnum])/(
                            CRmas[rindlr[0]]-SOGroupMass[rwnum])
                        SOGroupLen[-1]+=SOGroupLen[rwnum]
                        SOGroupOffset[-1]+=SOGroupLen[rwnum]
                        SOTotNids+=SOGroupLen[rwnum]
                SOTotNgroups+=1 
   
        else:
            Ncount+=1
        blp[rslpart]=False
        dens[rslpart]=-1*np.abs(dens[rslpart])

    SOGroupIDs=SOGroupIDs[:SOTotNids]
    ff=open(outfiles+"."+str(ii),'wb')
    d1=pack('q q q',SOTotNgroups,SOTotNids,bins3)
    ff.write(d1)
    ff.write(np.array(SOGroupLen,dtype=np.int32))
    ff.write(np.array(SOGroupOffset[:SOTotNgroups],dtype=np.uint32))
    ff.write(np.float32(SOGroupMass))
    ff.write(np.array(SOR500,dtype=np.float32))
    ff.write(np.array(SOpotPos,dtype=np.float32))
    ff.write(np.array(SOmcPos,dtype=np.float32))
    ff.write(np.array(SOPids))
    ff.write(SOGroupIDs)
    ff.close()

    return SOTotNgroups
