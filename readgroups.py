"""model read_SOVIR"""
import numpy as np
from struct import unpack

def readgroups(base,delta,endian=None,longid=None, sglnum=None, readid=None):
    """
    readfof(Base,delta,endian=None,longid=None)
             Read SO files from myoutput
     Parameters:
             Base: where your SO folder located
             delta: 500, 200, VIR, etc.
    	     little endian : ">", big endian : "<", other : "=" or "@"
             longid : If needed
	     sglnum : If specified, read only the file with this number
    return structures:
             TotNgroups,TotNids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs
    Example:
    --------
    TotNgroups,TotNids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs=readgroups("./","VIR",endian='=')
    --------
    updated time 1 March 2013 by wgcui
    """
    if endian==None:
        endian='='

    if sglnum!=None:
        fn=base+"/Groups_"+delta+"/SO"+"."+str(sglnum)
        npf=open(fn,'r')
        Ngroups,Nids,NTask=unpack(endian+'q q q',npf.read(8*3))
        buf=npf.read(4*Ngroups)
        GroupLen=np.ndarray(shape=Ngroups,dtype='int32',buffer=buf)
        #if longid == None:
        buf=npf.read(4*Ngroups)
        GroupOffset=np.ndarray(shape=Ngroups,dtype='uint32',buffer=buf)
        #else:
        #    buf=npf.read(8*Ngroups)
        #    GroupOffset=np.ndarray(shape=Ngroups,dtype='uint64',buffer=buf)
        buf=npf.read(4*Ngroups)
        GroupMass=np.ndarray(shape=Ngroups,dtype='float32',buffer=buf)
        buf=npf.read(4*Ngroups)
        GroupRad =np.ndarray(shape=Ngroups,dtype='float32',buffer=buf)
        buf=npf.read(4*3*Ngroups)
        GroupptPos=np.ndarray(shape=(Ngroups,3),dtype='float32',buffer=buf)
        buf=npf.read(4*3*Ngroups)
        GroupmcPos=np.ndarray(shape=(Ngroups,3),dtype='float32',buffer=buf)
        if (longid == None) | (longid == False):
            buf=npf.read(4*Ngroups)
            GroupPids=np.ndarray(shape=Ngroups,dtype='uint32',buffer=buf)
            if readid != None:
                buf=npf.read(4*Nids)
                GroupIDs=np.ndarray(shape=Nids,dtype='uint32',buffer=buf)
        else:
            buf=npf.read(8*Ngroups)
            GroupPids=np.ndarray(shape=Ngroups,dtype='uint64',buffer=buf)
            if readid != None:
                buf=npf.read(8*Nids)
                GroupIDs=np.ndarray(shape=Nids,dtype='uint64',buffer=buf)
        npf.close()
        SONgroups,SONids=Ngroups,Nids
    else:
        fnb,skip,NTask=0,0,1
        GroupLen   =np.array([],dtype='int32')
        GroupOffset=np.array([],dtype='uint32')
        GroupMass  =np.array([],dtype='float32')
        GroupRad   =np.zeros_like(GroupMass)
        GroupptPos =np.array([[],[],[]],dtype='float32')
        GroupptPos =GroupptPos.transpose()
        GroupmcPos =np.zeros_like(GroupptPos)
        if longid == None:
            GroupPids  =np.array([],dtype='uint32')
            #GroupIDs   =np.array([],dtype='uint32')
        else:
            GroupPids  =np.array([],dtype='uint64')
            #GroupIDs   =np.array([],dtype='uint64')
        GroupIDs   =[]
        SONgroups,SONids=0,0
        while fnb<NTask:
            fn=base+"/Groups_"+delta+"/SO"+"."+str(fnb)
            npf=open(fn,'r')
            Ngroups,Nids,NTask=unpack(endian+'q q q',npf.read(8*3))
            SONgroups+=Ngroups

            buf=npf.read(4*Ngroups)
            GroupLen=np.append(GroupLen,np.ndarray(shape=Ngroups,dtype='int32',buffer=buf))
            #if longid == None:
            buf=npf.read(4*Ngroups)
            GroupOffset=np.append(GroupOffset,np.ndarray(shape=Ngroups,dtype='uint32',buffer=buf)+SONids)
            #else:
            #    buf=npf.read(8*Ngroups)
            #    GroupOffset=np.append(GroupOffset,np.ndarray(shape=Ngroups,dtype='uint64',buffer=buf)+SONids)
            buf=npf.read(4*Ngroups) 
            GroupMass=np.append(GroupMass,np.ndarray(shape=Ngroups,dtype='float32',buffer=buf))
            buf=npf.read(4*Ngroups) 
            GroupRad =np.append(GroupRad,np.ndarray(shape=Ngroups,dtype='float32',buffer=buf))
            buf=npf.read(4*3*Ngroups)
            GroupptPos=np.append(GroupptPos,np.ndarray(shape=(Ngroups,3),dtype='float32',buffer=buf),axis=0)
            buf=npf.read(4*3*Ngroups)
            GroupmcPos=np.append(GroupmcPos,np.ndarray(shape=(Ngroups,3),dtype='float32',buffer=buf),axis=0)
            if (longid == None) | (longid == False):
                buf=npf.read(4*Ngroups)
                GroupPids=np.append(GroupPids,np.ndarray(shape=Ngroups,dtype='uint32',buffer=buf))
                if readid != None:
                    buf=npf.read(4*Nids)
                    #GroupIDs=np.append(GroupIDs,np.ndarray(shape=Nids,dtype='uint32',buffer=buf))
                    GroupIDs.extend(list(unpack(endian+str(Nids)+'L',buf)))
            else:
                buf=npf.read(8*Ngroups)
                GroupPids=np.append(GroupPids,np.ndarray(shape=Ngroups,dtype='uint64',buffer=buf))
                if readid != None:
                    buf=npf.read(8*Nids)
                    #GroupIDs=np.append(GroupIDs,np.ndarray(shape=Nids,dtype='uint64',buffer=buf))
                    GroupIDs.extend(list(unpack(endian+str(Nids)+'Q',buf)))

            fnb+=1
            SONids+=Nids
            npf.close()

    if endian!='=':
        GroupLen=GroupLen.byteswap()
        GroupOffset=GroupOffset.byteswap()
        GroupMass=GroupMass.byteswap()
        GroupRad=GroupRad.byteswap()
        GroupptPos=GroupptPos.byteswap()
        GroupmcPos=GroupmcPos.byteswap()
        GroupPids=GroupPids.byteswap()
        #if readid != None:
            #GroupIDs  =GroupIDs.byteswap()

    if longid == None:
        if readid != None:
            GroupIDs  = np.array(GroupIDs,dtype='uint32')
    else:
        GroupIDs  = np.array(GroupIDs,dtype='uint64')

    if readid != None:
        return(SONgroups,SONids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs)
    else:
        return(SONgroups,SONids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids)
