# Function to generate channel coefficients

'''
#-------------------- description of variables ----------------------
Nt:         number of BS transmit antennas 
K:          number of single-antenna users 
nIRSrow:    number of rows of IRS elements 
nIRScol:    number of columns of IRS elements 
locU:       array of users' locations 
Lambda:     carrier wavelength   
kappa:      Rician factor 
xt:         x-coordiate of the center of tx ULA 
yt:         y-coordiate of the center of tx ULA 
zt:         z-coordiate of the center of tx ULA 
xs:         x-coordiate of the center of IRS UPA
ys:         y-coordiate of the center of IRS UPA 
zs:         z-coordiate of the center of IRS UPA
locT:       array of coordinates of the tx antennas 
locS:       array of coordinates of IRS elements 
dTU:        array of distance between tx antennas and user's antenna
dSU:        array of distance between IRS elements and user's antenna 
dTS:        array of distance between tx antennas and IRS elements 
alphaDir:   pathloss exponent for direct links 
alphaIRS:   pathloss exponent for IRS-related links 
betaTU:     pathloss for BS-users' links 
betaTS:     pathloss for BS-IRS links 
betaSU:     pathloss for IRS-user links 
hTU_LoS:    LoS conponent for BS-user links 
hTU_NLoS:   NLoS component for BS-user links 
hTU:        stack of tx-users' channel vectors 
hTS_LoS:    LoS conponent for BS-IRS links 
hTS_NLoS:   NLoS component for BS-IRS links 
hTS:        BS-IRS channel matrix 
hSU_LoS:    LoS conponent for IRS-user links 
hSU_NLoS:   NLoS component for IRS-user links 
hSU:        stack of IRS-users' channel vectors 
Gt:         transmit-antenna gain 
Gr:         receive-antenna gain
#--------------------------------------------------------------------            
'''


import numpy as np
def ChanGen(Nt,K,nIRSrow,nIRScol,locU,Lambda):
    halfLambda = 0.5*Lambda
    quarterLambda = 0.25*Lambda
    kappa = 1
    
    
    #=========== Location of nodes/antennas/tiles (all in m)
    #----------- tx uniform linear array (ULA)
    xt = 0 
    yt = 20
    zt = 10
    #----------- IRS uniform planar array (UPA)
    xs = 30
    ys = 0
    zs = 5    
    
    #================ transmit antenna coordinates
    locTcenter = np.array([xt,yt,zt],dtype=float)
    locT = np.tile(locTcenter, (Nt, 1))
    if np.mod(Nt,2) == 0:
        locT[0,1] = yt-0.5*(Nt-2)*halfLambda - quarterLambda
    else:
        locT[0,1] = yt-0.5*(Nt-1)*halfLambda
    locT[:,1] = [locT[0,1]+nt*halfLambda for nt in range(Nt)]  
                    
    #================ IRS coordinates
    locIRScenter = np.array([xs,ys,zs],dtype=float)
    locS = np.tile(locIRScenter,(nIRSrow,nIRScol,1))
    if np.mod(nIRScol,2) == 0:
        locS[:,:,0] = xs - 0.5*(nIRScol-2)*halfLambda - quarterLambda
    else:
        locS[:,:,0] = xs - 0.5*(nIRScol-1)*halfLambda 
        
    locS[:,:,0] = [[locS[nRow,0,0]+nCol*halfLambda \
                     for nCol in range(nIRScol)] \
                     for nRow in range(nIRSrow)]
                    
    if np.mod(nIRSrow,2) == 0:
        locS[:,:,2] = zs - 0.5*(nIRSrow-2)*halfLambda - quarterLambda
    else:
        locS[:,:,2] = zs - 0.5*(nIRSrow-1)*halfLambda 
        
    locS[:,:,2] = [[locS[0,nCol,2]+nRow*halfLambda \
                     for nCol in range(nIRScol)] \
                     for nRow in range(nIRSrow)]     
    locS = np.reshape(locS,(nIRSrow*nIRScol,3)) 
    
    #================ calculating the distance between antennas/tiles
    dTU = np.array([np.linalg.norm(locU[k,:]-locT,axis=1) for k in range(K)])
    dSU = np.array([np.linalg.norm(locU[k,:]-locS,axis=1) for k in range(K)])
    dTS = np.transpose(np.array([np.linalg.norm(locT[nt,:]-locS,axis=1) for nt in range(Nt)]))    
                    
    #================ tx-user channels
    alphaDir = 3
    betaTU = ((4*np.pi/Lambda)**2)*(dTU**alphaDir)
    hTU_LoS = np.exp(-1j*2*np.pi*dTU/Lambda)
    hTU_NLoS = np.sqrt(1/2)*(np.random.randn(K,Nt)+1j*np.random.randn(K,Nt))
    hTU = np.sqrt((betaTU**(-1))/(kappa+1))*(np.sqrt(kappa)*hTU_LoS+hTU_NLoS)    
    
    #================ tx-IRS channels
    alphaIRS = 2
    Gt = 2
    cosGammaT = yt/dTS
    betaTS = ((4/Lambda)**2)*(dTS**alphaIRS)/(Gt*cosGammaT)
    HTS_LoS = np.exp(-1j*2*np.pi*dTS/Lambda)
    HTS_NLoS = np.sqrt(1/2)*(np.random.randn(nIRSrow*nIRScol,Nt)\
                                 +1j*np.random.randn(nIRSrow*nIRScol,Nt))
    HTS = np.sqrt((betaTS**(-1))/(kappa+1))*(np.sqrt(kappa)*HTS_LoS+HTS_NLoS)                        
                        
    #================ IRS-user channels                    
    Gr = 2
    cosGammaR = np.array([locU[k,1]/dSU[k,:] for k in range(K)])
    betaSU = ((4*np.pi/Lambda)**2)*(dSU**alphaIRS)/(Gr*cosGammaR) 
    hSU_LoS = np.exp(-1j*2*np.pi*dSU/Lambda)
    hSU_NLoS = np.sqrt(1/2)*(np.random.randn(K,nIRSrow*nIRScol)\
                                 +1j*np.random.randn(K,nIRSrow*nIRScol))
    hSU = np.sqrt((betaSU**(-1))/(kappa+1))*(np.sqrt(kappa)*hSU_LoS+hSU_NLoS)
    
    return hTU,HTS,hSU    