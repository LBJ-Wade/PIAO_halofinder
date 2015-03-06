import numpy as np
from struct import unpack

def readgroups(base,delta,endian=None,longid=None, sglnum=None, readid=None):
    """
    readgroups(Base,delta,endian=None,longid=None, sglnum=None, readid=None)
             Read SO files from output of PIAO

     Parameters:
             Base: where your SO folder located
             delta: The name of SO folder. They are normally named as this order:
                    “Groups_”+ overdensity + “_” + exts from snapshot + “_” + bin size in Mpc + “_” + SPH neighbours + “_nl”  or none
                    nl here means no overlapping, without nl means the SO groups can overlap.
    	     little endian : ">", big endian : "<", default : "=" or "@"
             longid : True, if you particle ID is in longlong.
	     sglnum : file number, if specified, read only the file with this number. Default, read all the output SO files.
             readid : True, if you want to read the particle IDs.

    return structures:
             TotNgroups,TotNids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs

             TotNgroups:     integer, the total number of groups 
             TotNids:        integer, the total number of particle IDs.
             GroupLen:       1-D int array with length of TotNgroups, the number of particles inside each group.
             GroupOffset:    1-D int array with length of TotNgroups +1, the offsets for the GroupLen.
             GroupMass:      1-D float array with length of TotNgroups, the mass for each group in unit of 10^10 M_sun. 
                             !Note!, there maybe some minus values for this mass, you have to omit them. 
                             These groups with minus values mean that they are only a substructure of a large group (by
                             means of their centre is located within the radius of other group). 
                             But they are identified first, when we have another group include them,
                             we count them in the large group and omit them in the first place.
            GroupRad:        1-D float array, the radii of each group
            GroupptPos:      2-D float array with shape of (TotNgroups,3), the centre of each group. 
                             Maximum SPH density centre, which is used for identifying SO groups.
            GroupmcPos:      2-D float array with shape of (TotNgroups,3), the centre of mass for each group.
            GroupPids:       1-D int/long (depend on the simulation) array with length of TotNgroups, 
                             the most dens particle’s ID for each group. GroupptPos is the position of this particle.
            GroupIDs:        1-D int/long  (depend on the simulation) array with length of TotNids. The IDs of particles for all groups.

    Example:
    --------
    TotNgroups,TotNids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs=readgroups("./","VIR",endian='=', readid=True)
    --------
    updated time 6 March 2015 by wgcui
    """

    if endian==None:
        endian='='

    if sglnum!=None:
        fn=base+delta+"/SO"+"."+str(sglnum)
        npf=open(fn,'r')
        Ngroups,Nids,NTask=unpack(endian+'q q q',npf.read(8*3))
        buf=npf.read(4*Ngroups)
        GroupLen=np.ndarray(shape=Ngroups,dtype='int32',buffer=buf)
        buf=npf.read(4*Ngroups)
        GroupOffset=np.ndarray(shape=Ngroups,dtype='uint32',buffer=buf)
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
        else:
            GroupPids  =np.array([],dtype='uint64')
        GroupIDs   =[]
        SONgroups,SONids=0,0
        while fnb<NTask:
            fn=base+delta+"/SO"+"."+str(fnb)
            npf=open(fn,'r')
            Ngroups,Nids,NTask=unpack(endian+'q q q',npf.read(8*3))
            SONgroups+=Ngroups

            buf=npf.read(4*Ngroups)
            GroupLen=np.append(GroupLen,np.ndarray(shape=Ngroups,dtype='int32',buffer=buf))
            buf=npf.read(4*Ngroups)
            GroupOffset=np.append(GroupOffset,np.ndarray(shape=Ngroups,dtype='uint32',buffer=buf)+SONids)
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
                    GroupIDs.extend(list(unpack(endian+str(Nids)+'L',buf)))
            else:
                buf=npf.read(8*Ngroups)
                GroupPids=np.append(GroupPids,np.ndarray(shape=Ngroups,dtype='uint64',buffer=buf))
                if readid != None:
                    buf=npf.read(8*Nids)
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

    if longid == None:
        if readid != None:
            GroupIDs  = np.array(GroupIDs,dtype='uint32')
    else:
        GroupIDs  = np.array(GroupIDs,dtype='uint64')

    if readid != None:
        return(SONgroups,SONids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids,GroupIDs)
    else:
        return(SONgroups,SONids,GroupLen,GroupOffset,GroupMass,GroupRad,GroupptPos,GroupmcPos,GroupPids)
