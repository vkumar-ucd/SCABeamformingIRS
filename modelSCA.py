import numpy as np
from mosek.fusion import *
import time

#============== Hermitian function
def Herm(x):
    return x.conj().T
    

#============== power minimization function
def minimizePower(Nt,K,Ns,thetaVecCurrent,wCurrent,thetaVecNormSqCurrent,gCurrent,Xi,hTU,hSU,HTS,gamma):
        
    #------ MOSEK model
    minPowerModel = Model()

    #------------- variables --------------------
    wR = minPowerModel.variable('wR', [Nt,K], Domain.unbounded()) # real component of variable w
    wI = minPowerModel.variable('wI', [Nt,K], Domain.unbounded()) # imaginary component of variable w 
    thetaR = minPowerModel.variable('thetaR',[Ns,1],Domain.unbounded()) # real component of variable theta
    thetaI = minPowerModel.variable('thetaI',[Ns,1],Domain.unbounded()) # imaginary component of variable theta
    t = minPowerModel.variable('t', [K,K], Domain.unbounded()) # variable t
    tBar = minPowerModel.variable('tBar', [K,K], Domain.unbounded()) # variable tBar 
    tObj = minPowerModel.variable('tObj',1,Domain.unbounded()) # variable tObj

    wRTranspose = wR.transpose()
    wITranspose = wI.transpose()

    #------ channels 
    gR = Expr.sub(Expr.sub(Expr.sub(Expr.add(hTU.real,\
                                             Expr.mul(Expr.mulElm(hSU.real,Expr.transpose(Expr.repeat(thetaR,K,1))),HTS.real)),\
                                    Expr.mul(Expr.mulElm(hSU.real,Expr.transpose(Expr.repeat(thetaI,K,1))),HTS.imag)),\
                           Expr.mul(Expr.mulElm(hSU.imag,Expr.transpose(Expr.repeat(thetaR,K,1))),HTS.imag)),\
                  Expr.mul(Expr.mulElm(hSU.imag,Expr.transpose(Expr.repeat(thetaI,K,1))),HTS.real)) 
                                        # real component of the effective BS-User channel

    gRTranspose = Expr.transpose(gR) # transpose of gR 

    gI = Expr.sub(Expr.add(Expr.add(Expr.add(hTU.imag,\
                                             Expr.mul(Expr.mulElm(hSU.real,Expr.transpose(Expr.repeat(thetaR,K,1))),HTS.imag)),\
                                    Expr.mul(Expr.mulElm(hSU.real,Expr.transpose(Expr.repeat(thetaI,K,1))),HTS.real)),\
                           Expr.mul(Expr.mulElm(hSU.imag,Expr.transpose(Expr.repeat(thetaR,K,1))),HTS.real)),\
                  Expr.mul(Expr.mulElm(hSU.imag,Expr.transpose(Expr.repeat(thetaI,K,1))),HTS.imag))
                                        # imaginary component of the effective BS-User channel

    gITranspose = Expr.transpose(gI) # transpose of gI

    #------ calculating constants 
    aCurrent,aAbsSQ,bCurrentR,bCurrentI,\
            bCurrentNormSQ,delta1,delta2 = ComputeParametersSet1(K,gCurrent,wCurrent)  
    aR = aCurrent.real
    aI = aCurrent.imag    

    #------------- objective --------------------
    obj = Expr.sub(tObj,Expr.mul(Xi,Expr.sub(Expr.add(Expr.dot(2*thetaVecCurrent.real,thetaR),\
                                                      Expr.dot(2*thetaVecCurrent.imag,thetaI)),\
                                             thetaVecNormSqCurrent)))
    minPowerModel.objective("obj",ObjectiveSense.Minimize,obj) # objective function

    #------------- constraints --------------------

    minPowerModel.constraint(Expr.vstack(Expr.mul(Expr.add(tObj,1),0.5),Expr.flatten(Expr.hstack(wR,wI)),\
                             Expr.mul(Expr.sub(tObj,1),0.5)),Domain.inQCone())  
                                            # constraint for the slack variable tObj

    minPowerModel.constraint(Expr.hstack(Expr.constTerm(Matrix.ones(Ns,1)),\
                                         Expr.hstack(thetaR,thetaI)), Domain.inQCone()) 
                                            # relaxed unit-modulus constraints                                 

    for k in range(K):
        #--------------- constraint (15b)
        lhsB = Expr.mul(Expr.sub(Expr.add(Expr.add(Expr.add(Expr.dot(delta1[k,:],gR.slice([k,0],[k+1,Nt])),\
                                                            Expr.dot(delta2[k,:],gI.slice([k,0],[k+1,Nt]))),\
                                                   Expr.dot(bCurrentR[:,k].T,wR.slice([0,k],[Nt,k+1]))),\
                                          Expr.dot(bCurrentI[:,k].T,wI.slice([0,k],[Nt,k+1]))),\
                                 0.5*bCurrentNormSQ[k]+aAbsSQ[k]), 1/(2*gamma))
        
        rhsB = Expr.hstack(Expr.hstack(Expr.hstack(Expr.hstack(Expr.reshape(t.pick([[k,j] for j in range(K) if j!=k]),1,K-1),\
                                                   Expr.reshape(tBar.pick([[k,j] for j in range(K) if j!=k]),1,K-1)),\
                                                   Expr.mul(Expr.sub(Expr.add(Expr.mul(gR.slice([k,0],[k+1,Nt]),aR[k]),\
                                                                              Expr.mul(gI.slice([k,0],[k+1,Nt]),aI[k])),\
                                                                     wRTranspose.slice([k,0],[k+1,Nt])),\
                                                            np.sqrt(1/(2*gamma)))),\
                                       Expr.mul(Expr.sub(Expr.sub(Expr.mul(gR.slice([k,0],[k+1,Nt]),aI[k]),\
                                                                  Expr.mul(gI.slice([k,0],[k+1,Nt]),aR[k])),\
                                                         wITranspose.slice([k,0],[k+1,Nt])),\
                                                np.sqrt(1/(2*gamma)))),\
                           Expr.sub(lhsB,1))
        
        minPowerModel.constraint(Expr.hstack(lhsB,rhsB),Domain.inQCone())

        #--------------- compute another set of parameters 

        Lambda,LambdaTilde,normCSQ,eta,etaTilde,normDSQ,\
        psi,psiTilde,normESQ,phi,phiTilde,normFSQ\
                = ComputeParametersSet2(k,K,gCurrent,wCurrent)   

        #--------------- constraint (10)     

        lhsC = Expr.add(Expr.add(Expr.reshape(t.pick([[k,l] for l in range(K) if l!= k]),K-1,1),\
                                 Expr.reshape(Expr.mulDiag(0.5*Lambda,\
                                              Expr.sub(Expr.repeat(gRTranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                       Expr.transpose(Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if ((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1)),\
                       Expr.reshape(Expr.mulDiag(0.5*LambdaTilde,\
                                                 Expr.add(Expr.repeat(gITranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                Expr.transpose(Expr.reshape(\
                Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if ((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1))

        rhsC = Expr.mul(Expr.sub(lhsC,1+0.25*normCSQ),0.5)
        lhsC = Expr.mul(Expr.add(lhsC,1-0.25*normCSQ),0.5)

        rhsC = Expr.hstack(Expr.hstack(rhsC,\
                                       Expr.mul(Expr.sub(\
                Expr.reshape(Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt),\
                                       Expr.repeat(gI.slice([k,0],[k+1,Nt]),K-1,0)),0.5)),\
                                       Expr.mul(Expr.add(\
                Expr.reshape(Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt),\
                                       Expr.repeat(gR.slice([k,0],[k+1,Nt]),K-1,0)),0.5))

        minPowerModel.constraint(Expr.hstack(lhsC,rhsC),Domain.inQCone()) # constraint in (10)

        #--------------- constraint (11)    

        lhsD = Expr.add(Expr.add(Expr.reshape(t.pick([[k,l] for l in range(K) if l!= k]),K-1,1),\
                                 Expr.reshape(Expr.mulDiag(0.5*eta,\
                                                           Expr.add(Expr.repeat(gRTranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                                    Expr.transpose(Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1)),\
                                 Expr.reshape(Expr.mulDiag(0.5*etaTilde,\
                                                           Expr.sub(Expr.repeat(gITranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                                    Expr.transpose(Expr.reshape(\
                Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1))

        rhsD = Expr.mul(Expr.sub(lhsD,1+0.25*normDSQ),0.5)
        lhsD = Expr.mul(Expr.add(lhsD,1-0.25*normDSQ),0.5)       

        rhsD = Expr.hstack(Expr.hstack(rhsD,\
                                       Expr.mul(Expr.add(\
                Expr.reshape(Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt),\
                                                         Expr.repeat(gI.slice([k,0],[k+1,Nt]),K-1,0)),0.5)),
                                       Expr.mul(Expr.sub(Expr.repeat(gR.slice([k,0],[k+1,Nt]),K-1,0),\
                                                         Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)),0.5))

        minPowerModel.constraint(Expr.hstack(lhsD,rhsD),Domain.inQCone()) # constraint in (11)

        #--------------- constraint (13)     

        lhsE = Expr.add(Expr.add(Expr.reshape(tBar.pick([[k,l] for l in range(K) if l!= k]),K-1,1),\
                                 Expr.reshape(Expr.mulDiag(0.5*psi,\
                                                           Expr.sub(Expr.repeat(gRTranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                                    Expr.transpose(Expr.reshape(\
                Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if ((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1)),\
                       Expr.reshape(Expr.mulDiag(0.5*psiTilde,\
                                                 Expr.sub(Expr.repeat(gITranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                          Expr.transpose(Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if ((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1))

        rhsE = Expr.mul(Expr.sub(lhsE,1+0.25*normESQ),0.5)
        lhsE = Expr.mul(Expr.add(lhsE,1-0.25*normESQ),0.5)

        rhsE = Expr.hstack(Expr.hstack(rhsE,\
                                       Expr.mul(Expr.add(Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt),\
                                                         Expr.repeat(gI.slice([k,0],[k+1,Nt]),K-1,0)),0.5)),\
                           Expr.mul(Expr.add(Expr.reshape(\
                Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt),\
                                    Expr.repeat(gR.slice([k,0],[k+1,Nt]),K-1,0)),0.5))

        minPowerModel.constraint(Expr.hstack(lhsE,rhsE),Domain.inQCone()) # constraint in (13)

        #--------------- constraint (14)

        lhsF = Expr.add(Expr.add(Expr.reshape(tBar.pick([[k,l] for l in range(K) if l!= k]),K-1,1),\
                                 Expr.reshape(Expr.mulDiag(0.5*phi,\
                                                           Expr.add(Expr.repeat(gRTranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                                    Expr.transpose(Expr.reshape(\
                Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if ((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1)),\
                        Expr.reshape(Expr.mulDiag(0.5*phiTilde,\
                                                  Expr.add(Expr.repeat(gITranspose.slice([0,k],[Nt,k+1]),K-1,1),\
                                                           Expr.transpose(Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if ((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)))),K-1,1))

        rhsF = Expr.mul(Expr.sub(lhsF,1+0.25*normFSQ),0.5)
        lhsF = Expr.mul(Expr.add(lhsF,1-0.25*normFSQ),0.5)

        rhsF = Expr.hstack(Expr.hstack(rhsF,\
                                       Expr.mul(Expr.sub(Expr.reshape(\
                Expr.flatten(wRTranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt),\
                                                Expr.repeat(gI.slice([k,0],[k+1,Nt]),K-1,0)),0.5)),\
                           Expr.mul(Expr.sub(Expr.repeat(gR.slice([k,0],[k+1,Nt]),K-1,0),\
                                             Expr.reshape(\
                Expr.flatten(wITranspose).pick([[l] for l in range(K*Nt) if((l<k*Nt ) or (l>=k*Nt+Nt))]),K-1,Nt)),0.5))

        minPowerModel.constraint(Expr.hstack(lhsF,rhsF),Domain.inQCone()) # constraint in (14)

    
    try:
        start = time.time()
        minPowerModel.solve()
        solTime = time.time()-start
        w = np.reshape(wR.level()+1j*wI.level(),(Nt,K))
        theta = thetaR.level()+1j*thetaI.level()
        minPowerModel.dispose()
        return w, theta, 1, solTime
    except SolutionError:
        return 0, 0, 0, 0
           
    
              
#============== function to calculate the first set of constants
def ComputeParametersSet1(K,gCurrent,wCurrent):  

    aCurrent = np.diag(gCurrent@wCurrent)
    bCurrent = np.transpose(np.array([aCurrent[k]*Herm(gCurrent[k,:])+wCurrent[:,k] for k in range(K)]))
    aR = aCurrent.real
    aI = aCurrent.imag
    aAbsSQ = abs(aCurrent)**2    
    bCurrentR = bCurrent.real
    bCurrentI = bCurrent.imag
    bCurrentNormSQ = np.linalg.norm(bCurrent,axis=0)**2 # columnwise 2-norm
    delta1 = np.array([bCurrentR[:,k].T*aR[k]\
                       +bCurrentI[:,k].T*aI[k] for k in range(K)])
    delta2 = np.array([bCurrentR[:,k].T*aI[k]\
                       -bCurrentI[:,k].T*aR[k] for k in range(K)])

    return aCurrent,aAbsSQ,bCurrentR,bCurrentI,bCurrentNormSQ,delta1,delta2


#============== function to calculate the second set of constants
def ComputeParametersSet2(k,K,gCurrent,wCurrent):
    
    #------------ constant values for (10) ------------         
    Lambda = np.real(np.tile(gCurrent[[k],:],(K-1,1)))-np.real(wCurrent[:,np.arange(K) != k].T)
    LambdaTilde = np.imag(np.tile(gCurrent[[k],:],(K-1,1)))+np.imag(wCurrent[:,np.arange(K) != k].T)
    normCSQ = np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k],:]),(1,K-1))\
                                            -wCurrent[:,np.arange(K) != k],axis=0)**2,(-1,1))

    #------------ constant values for (11) ------------
    eta = np.real(np.tile(gCurrent[[k],:],(K-1,1)))+np.real(wCurrent[:,np.arange(K) != k].T)
    etaTilde = np.imag(np.tile(gCurrent[[k],:],(K-1,1)))-np.imag(wCurrent[:,np.arange(K) != k].T)
    normDSQ = np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k],:]),(1,K-1))\
                                            +wCurrent[:,np.arange(K) != k],axis=0)**2,(-1,1))

    #------------ constant values for (13) ------------
    psi= np.real(np.tile(gCurrent[[k],:],(K-1,1)))-np.imag(wCurrent[:,np.arange(K) != k].T)
    psiTilde = np.imag(np.tile(gCurrent[[k],:],(K-1,1)))-np.real(wCurrent[:,np.arange(K) != k].T)
    normESQ=  np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k],:]),(1,K-1))\
                                            +1j*wCurrent[:,np.arange(K) != k],axis=0)**2,(-1,1))
    
    #------------ constant values for (14) ------------
    phi = np.real(np.tile(gCurrent[[k],:],(K-1,1)))+np.imag(wCurrent[:,np.arange(K) != k].T)
    phiTilde = np.imag(np.tile(gCurrent[[k],:],(K-1,1)))+np.real(wCurrent[:,np.arange(K) != k].T)
    normFSQ =  np.reshape(np.linalg.norm(np.tile(Herm(gCurrent[[k],:]),(1,K-1))\
                                            -1j*wCurrent[:,np.arange(K) != k],axis=0)**2,(-1,1))                                          

    return Lambda,LambdaTilde,normCSQ,eta,etaTilde,normDSQ,psi,psiTilde,normESQ,phi,phiTilde,normFSQ