from struct import *
from pylab import *
import string
import os

def readdp(path,fname=None,sglnum=None):

    """
    readdp(path,fname,sglnum=None)
             Read SO of density profiles
     Parameters:
             path: where your SO folder located
             fname: If specified, will change default name "DP.*" into fname.*
             sglnum : If specified, read only the file with this number
    return structures:
             gnum,bl,tmgid,mtid,rvir,tpd,(dmpd,stpd),(Tgnum)
    Example:
    --------
    gnum,bl,tmgid,mtid,rvir,tpd,dmpd,stpd=readdp("/scratch3/wgcui/SOgroups/AGN/Groups_VIR_059_41Mpc_64_nl")
    --------
    updated time 18 Sep 2013 by wgcui
    """

    if fname==None:
        fname="DP"

    if sglnum != None:
        f=open(path+"/"+fname+"."+str(sglnum),'rb')
        gnum,bl=unpack('i i',f.read(8))
        #tmgid=np.ndarray(shape=(gnum),dtype='int32',buffer=f.read(4*gnum))
        #mtid=np.ndarray(shape=(gnum),dtype='int32',buffer=f.read(4*gnum))
        rvir=np.ndarray(shape=(gnum),dtype='float32',buffer=f.read(4*gnum))
        tpd=np.ndarray(shape=(gnum,bl),dtype='float32',buffer=f.read(4*gnum*bl))
        idn=string.find(path, "/")
        if idn ==-1:
            dmpd=np.ndarray(shape=(gnum,bl),dtype='float32',buffer=f.read(4*gnum*bl))
            stpd=np.ndarray(shape=(gnum,bl),dtype='float32',buffer=f.read(4*gnum*bl))
            f.close()
            return gnum,bl,tmgid,mtid,rvir,tpd,dmpd,stpd
        else:
            f.close()
            return gnum,bl,tmgid,mtid,rvir,tpd

    else:
        Tgnum=[]
        Tmgid,Tmtid,Trvir,Ttpd=[],[],[],[]
        idn=string.find(path, "/")
        if idn==-1:
            Tdmpd,Tstpd=[],[]
        for i in range(10000):
            fn=path+"/"+fname+"."+str(i)
            if os.path.isfile(fn):
                f=open(fn,'rb')
                gnum,bl=unpack('i i',f.read(8))
                tmgid=np.ndarray(shape=(gnum),dtype='int32',buffer=f.read(4*gnum))
                mtid=np.ndarray(shape=(gnum),dtype='int32',buffer=f.read(4*gnum))
                rvir=np.ndarray(shape=(gnum),dtype='float32',buffer=f.read(4*gnum))
                tpd=np.ndarray(shape=(gnum,bl),dtype='float32',buffer=f.read(4*gnum*bl))
                Tgnum.extend([gnum])
                Tmgid.extend(tmgid.tolist())
                Tmtid.extend(mtid.tolist())
                Trvir.extend(rvir.tolist())
                Ttpd.extend(tpd.tolist())
                if idn==-1:
                    dmpd=np.ndarray(shape=(gnum,bl),dtype='float32',buffer=f.read(4*gnum*bl))
                    stpd=np.ndarray(shape=(gnum,bl),dtype='float32',buffer=f.read(4*gnum*bl))
                    Tdmpd.extend(dmpd.tolist())
                    Tstpd.extend(stpd.tolist())
                f.close()
            else:
                print "Please check if ", i, " is your final file or not!"
                break

        Tgnum=np.array(Tgnum)
        if idn==-1:
            return Tgnum.sum(),bl,np.array(Tmgid),np.array(Tmtid),np.array(Trvir),np.array(Ttpd),np.array(Tdmpd),np.array(Tstpd),Tgnum
        else:
            return Tgnum.sum(),bl,np.array(Tmgid),np.array(Tmtid),np.array(Trvir),np.array(Ttpd),Tgnum
