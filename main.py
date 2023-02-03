# A novel SCA-based method for beamforming optimization in IRS/RIS-assisted MU-MISO downlink
# Authors: (*)Vaibhav Kumar, (^)Rui Zhang, ($)Marco Di Renzo, and (*)Le-Nam Tran
# DOI: 10.1109/LWC.2022.3224316
# Journal: IEEE Wireless Communications Letters
# (*): School of Electrical and Electronic Engineering, University College Dublin, D04 V1W8 Dublin, Ireland
# (^): The Chinese University of Hong Kong, Shenzhen, and Shenzhen Research Institute of Big Data, Shenzhen, China 518172
# (^): Department of Electrical and Computer Engineering, National University of Singapore, Singapore 117583
# ($): Universite Paris-Saclay, CNRS, CentraleSupelec, Laboratoire des Signaux et Systemes, 3 Rue Joliot-Curie, 91192 Gif-sur-Yvette, France
# email: vaibhav.kumar@ieee.org / vaibhav.kumar@ucd.ie 

'''
#-------------------- description of variables ----------------------
xu:                 x-coordinate of the center of users
yu:                 y-coordinate of the center of users
zu:                 z-coordinate of the center of users
userRadius:         radius of the user circle
x1:                 x-coordinate of the first user 
y1:                 y-coordinate of the first user 
z1:                 z-coordinate of the first user 
xCandidate:         x-coordinate of the candidate user 
yCandidate:         y-coordinate of the candidate user 
zCandidate:         z-coordinate of the candidate user 
candidateUserLoc:   location of the candidate user 
disCandidateUser:   distance of the candidate user from existing user 
minDis:             minimum distance between the users
LocU/locU           numpy array to hold user locations 
Nt:                 number of BS transmit antennas 
nIRSrow:            number of rows of IRS elements 
nIRScol:            number of columns of IRS elements 
K:                  number of single-antenna users 
gamma:              target SINR 
f:                  carrier center frequency in Hz
c:                  speed of light in m/s 
Lambda:             carrier wavelength in m
N0:                 noise PSD
B:                  bandwidth 
sigma:              standard deviation of AWGN noise 
epsilon:            convergence tolerance 
Xi:                 regularization parameter 
thetaVecCurrent:    current IRS reflection vector 
thetaMatCurrent:    current IRS reflection matrix 
relChange:          relative change in the objective 
iIter:              iteration counter 
hTU:                stack of BS-users channel vectors
HTS:                BS-IRS channle matrix 
hSU:                stack of IRS-users channel vectors 
gCurrent:           stack of current effective (direct+cascaded) BS-users chennel vectors 
wCurrent:           stack of current transmit beamforming vectors
objSeqW:            numpy array to hold objective sequence in Watts 
objSeqdBm:          numpy array to hold objective sequence in dBm
#--------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
from channelModel import ChanGen
from mosek.fusion import *
from modelSCA import *
np.random.seed(1) 

#============== Hermitian function
def Herm(x):
    # returns the conjugate transpose 
    return x.conj().T

#============== db2pow function
def db2pow(x):
    # returns the dB value 
    return 10**(0.1*x)

#============== pow2db function
def pow2db(x):
    # returns the power value
    return 10*np.log10(x)

#============== function to generate user locations
def GenerateUserLocations():
    # returns the coordiate locations of users
    xu = 350
    yu = 10
    zu = 2.0
    userRadius = 5
    randRadius = userRadius*np.random.rand(1)
    randAngle = np.random.rand(1)
    x1 = xu+randRadius*np.cos(2*np.pi*randAngle)
    y1 = yu+randRadius*np.sin(2*np.pi*randAngle)
    z1 = np.array([zu],dtype=float)
    LocU = np.array([x1,y1,z1])
    counter = 1
    while counter < K:
        randRadius = userRadius*np.random.rand(1)
        randAngle = np.random.rand(1)
        xCandidate = xu+randRadius*np.cos(2*np.pi*randAngle)
        yCandidate = yu+randRadius*np.sin(2*np.pi*randAngle)
        zCandidate = np.array([zu],dtype=float)
        candidateUserLoc = np.array([xCandidate,yCandidate,zCandidate])
        disCandidateUser = np.linalg.norm(candidateUserLoc-LocU,axis=0)
        minDis = np.amin(disCandidateUser)
        if minDis >= 2*Lambda:
            LocU = np.append(LocU,candidateUserLoc,axis=1)
            counter = counter+1
    LocU = LocU.T
    return LocU   

def initialW(hHermR,hHermI): 
    wModel = Model()        
    wR = wModel.variable('wR', [Nt,K], Domain.unbounded() )
    wI = wModel.variable('wI', [Nt,K], Domain.unbounded() )
    t = wModel.variable('t',1, Domain.unbounded())
    F = Expr.vstack(wR, wI)
    y = Expr.vstack(t,Expr.flatten(F))
    wModel.objective("obj", ObjectiveSense.Minimize, t)
    wModel.constraint("qc1", y, Domain.inQCone())   
    for k in range(K):
        u1 = Expr.mul(hHermR[k,:],wI.slice([0,k],[Nt,k+1]))
        u2 = Expr.mul(hHermI[k,:],wR.slice([0,k],[Nt,k+1]))
        wModel.constraint(Expr.add(u1,u2), Domain.equalsTo(0.0))        
        u1 = Expr.mul(hHermR[k,:],wR.slice([0,k],[Nt,k+1]))
        u2 = Expr.mul(hHermI[k,:],wI.slice([0,k],[Nt,k+1]))
        y1 = Expr.flatten(Expr.sub(u1,u2))        
        y2 = Expr.constTerm(1)
        for j in range(K):
            if j != k:
                a = Expr.mul(hHermR[k,:],wR.slice([0,j],[Nt,j+1]))
                b = Expr.mul(hHermI[k,:],wI.slice([0,j],[Nt,j+1]))              
                y2 = Expr.hstack(y2,Expr.sub(a,b))
                a = Expr.mul(hHermI[k,:],wR.slice([0,j],[Nt,j+1]))
                b = Expr.mul(hHermR[k,:],wI.slice([0,j],[Nt,j+1]))
                y2 = Expr.hstack(y2,Expr.add(a,b))
        y2 = Expr.mul(np.sqrt(gamma),Expr.flatten(y2) )       
        wModel.constraint(Expr.vstack(y1,y2), Domain.inQCone())   
    wModel.solve()
    w = np.reshape(wR.level()+1j*wI.level(),(Nt,K))
    wModel.dispose()
    return w    

def printStatus():
    sinrNum = abs(np.sum(gCurrent.T*wCurrent,0))**2
    sinrDen = 1+np.sum(abs(np.einsum('ij,jk->ik',gCurrent,wCurrent))**2,1)-sinrNum
    print(f"Target SINR = {pow2db(gamma):.3f} dB")
    sinr = sinrNum/sinrDen
    for k in range(K):
        print(f"User {k} SINR = {pow2db(sinr[k]):.3f} dB")
    print(f"Required transmit power = {objSeq[-1]:.3f} W")
    print(f"Required transmit power = {pow2db(objSeq[-1])+30:.3f} dBm")
    return[] 


#============== System parameters
Nt = 4
nIRSrow = 12
nIRScol = nIRSrow
Ns = nIRSrow*nIRScol
K = 4
gamma = db2pow(20)
f = 2e9
c = 3e8
Lambda = c/f
N0 = db2pow(-174-30)
B = 20e6
sigma = np.sqrt(B*N0)
epsilon = 1e-3
Xi = 0.001
relChange = 1e3
iIter = 0
objSeq = []

#============= Generating user antenna coordinates
locU = GenerateUserLocations()

#============= IRS beamforming vector initialization  
thetaVecCurrent = np.ones(Ns,dtype=complex)
thetaMatCurrent = np.diag(thetaVecCurrent)

#============= Channel generation and normalization 
hTU,HTS,hSU = ChanGen(Nt,K,nIRSrow,nIRScol,locU,Lambda)
hTU = (1/sigma)*hTU 
hSU = (1/sigma)*hSU
gCurrent = hTU+hSU@thetaMatCurrent@HTS

#============= Transmit beamformer initialization
wCurrent = initialW(gCurrent.real,gCurrent.imag)
objSeq.append(np.linalg.norm(wCurrent,'fro')**2)      

#============= implementing the SCA-based algorithm 
while relChange > epsilon:
    iIter = iIter+1  
    #=========== optimize w and thetaVec
    thetaVecNormSqCurrent = np.linalg.norm(thetaVecCurrent)**2
    wCurrent,thetaVecCurrent,solFlag,solTime\
        = minimizePower(Nt,K,Ns,thetaVecCurrent,wCurrent,thetaVecNormSqCurrent,gCurrent,Xi,hTU,hSU,HTS,gamma)
    if solFlag == 1:
        #=========== update channels
        thetaMatCurrent = np.diag(thetaVecCurrent)
        gCurrent = hTU+hSU@thetaMatCurrent@HTS 
        objSeq.append(np.linalg.norm(wCurrent,'fro')**2)     
        print('===================================')
        print('Iteration number = ',iIter)
        printStatus()
    else:
        print('Bad channel! Problem cannot be solved optimally. Try another seed!')
        break 
    if len(objSeq) > 3:
        relChange = (objSeq[-2] - objSeq[-1])/objSeq[-1]
if solFlag == 1:
    print('-------------FINAL RESULTS-----------------')
    printStatus()
    plt.plot(pow2db(objSeq)+30)
    plt.xlabel('Iteration number',fontsize=15)
    plt.ylabel('Required transmit power (dBm)',fontsize=15)
    plt.show()