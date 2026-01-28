#Based on the appendices provided in the dissertation, 
#here is the full Python code for the **pyTruss** 
#structural analysis program and the optimization algorithms (**DE** and **CaDE**).

### Appendix A: pyTruss Source Code

#### A.1. Module pyTruss2D
#This module is used for analyzing 2D planar trusses.

#```python
import numpy as np

#material properties
E = 70000 #N/mm2
A = 300 #mm2

#node coordinates
nodeCords = np.array([,
                      ,
                      ,
                      ,
                      ,
                      ])

#member connectivities
elemNodes = np.array([,
                     ,
                     ,
                     ,
                     ,
                     ,
                     ,
                     ,
                     ,
                     ,
                     ])

numElem = elemNodes.shape
numNodes = nodeCords.shape
tdof = 2*numNodes
xx = nodeCords[:,0]
yy = nodeCords[:,1]

#force
F = np.zeros((tdof,1))
F = -50000 #N
F = -100000 #N
F = -50000 #N

#boundary conditions
presDof=np.array()
actDof=np.setdiff1d(np.arange(tdof),presDof)

#global stiffness matrix
stiffness=np.zeros((tdof,tdof))
L = np.zeros((numElem,)) #lengths of members
c = np.zeros((numElem,))
s = np.zeros((numElem,))

for e in range(numElem):
    indice = elemNodes[e,:]
    elemDof = np.array([indice*2,indice*2+1,indice*2,indice*2+1])
    xa = xx[indice]-xx[indice]
    ya = yy[indice]-yy[indice]
    L[e] = np.sqrt(xa*xa+ya*ya)
    c[e] = xa/L[e]
    s[e] = ya/L[e]
    
    EA = E*A
    k1=(EA/L[e])* np.array([[ c[e]*c[e], c[e]*s[e],-c[e]*c[e],-c[e]*s[e]],
                            [ c[e]*s[e], s[e]*s[e],-c[e]*s[e],-s[e]*s[e]],
                            [-c[e]*c[e],-c[e]*s[e], c[e]*c[e], c[e]*s[e]],
                            [-c[e]*s[e],-s[e]*s[e], c[e]*s[e], s[e]*s[e]]])
    stiffness[np.ix_(elemDof,elemDof)]+=k1

#displacements
U=np.zeros((tdof,1))
disp=np.linalg.solve(stiffness[np.ix_(actDof,actDof)],F[np.ix_(actDof)]);
U[np.ix_(actDof)]=disp

#stress
S=np.zeros((numElem,1))
for e in range(numElem):
    indice= elemNodes[e,:]
    elemDof=np.array([indice*2, indice*2+1, indice*2, indice*2+1 ])
    S[e]=(E/L[e])*np.dot(np.array([-c[e],-s[e],c[e],s[e]]),U[np.ix_(elemDof)])
```

#### A.2. Module pyTruss3D
This module is used for analyzing 3D space trusses.

```python
import numpy as np

#material properties
E = 70000 #N/mm2
A = 300 #mm3

#node coordinates
nodeCords = np.array([, #node1
                      , #node2
                      , #node3
                      , #node4
                      , #node5
                      , #node6
                      , #node7
                      , #node8
                      ]) #node9

#member connectivites
elemNodes = np.array([,  #member1-node1-node5
                     ,  #member2-node2-node6
                     ,  #member3-node3-node7
                     ,  #member4-node4-node8
                     ,  #member5-node5-node2
                     ,  #member6-node2-node7
                     ,  #member7-node5-node4
                     ,  #member8-node8-node3
                     ,  #member9-node5-node6
                     ,  #member10-node6-node7
                     ,  #member11-node7-node8
                     ,  #member12-node8-node5
                     ,  #member13-node6-node9
                     ,  #member14-node7-node9
                     ]) #member15-node8-node9

numElem = elemNodes.shape
numNodes = nodeCords.shape
tdof = 3*numNodes
xx = nodeCords[:,0]
yy = nodeCords[:,1]
zz = nodeCords[:,2]

#force
F = np.zeros((tdof,1))
F = 70710.6781186548
F = 70710.6781186548

#boundary conditions
presDof=np.array()
actDof=np.setdiff1d(np.arange(tdof),presDof)

#global stiffness matrix
stiffness=np.zeros((tdof,tdof))
L = np.zeros((numElem,)) #lengths of members
cx = np.zeros((numElem,))
cy = np.zeros((numElem,))
cz = np.zeros((numElem,))

for e in range(numElem):
    indice= elemNodes[e,:]
    elemDof=np.array([indice*3, indice*3+1, indice*3+2,
                      indice*3, indice*3+1, indice*3+2])
    xa = xx[indice]-xx[indice]
    ya = yy[indice]-yy[indice]
    za = zz[indice]-zz[indice]
    L[e] = np.sqrt(xa*xa+ya*ya+za*za)
    cx[e] = xa/L[e]
    cy[e] = ya/L[e]
    cz[e] = za/L[e]
    
    T =np.array([[cx[e]*cx[e],cx[e]*cy[e],cx[e]*cz[e]],
                 [cy[e]*cx[e],cy[e]*cy[e],cy[e]*cz[e]],
                 [cz[e]*cx[e],cz[e]*cy[e],cz[e]*cz[e]]])
    T1 = np.concatenate(( T,-T),axis=1)
    T2 = np.concatenate((-T, T),axis=1)
    
    EA = E*A
    k1 = (EA/L[e]) * np.concatenate((T1, T2),axis=0)
    stiffness[np.ix_(elemDof,elemDof)]+=k1

#displacements
U = np.zeros((tdof,1))
disp = np.linalg.solve(stiffness[np.ix_(actDof,actDof)],F[np.ix_(actDof)]);
U[np.ix_(actDof)] = disp

#stress
S = np.zeros((numElem,1))
for e in range(numElem):
    indice= elemNodes[e,:]
    elemDof = np.array([indice*3, indice*3+1, indice*3+2,
                        indice*3, indice*3+1, indice*3+2])
    S[e]=(E/L[e])*np.dot(np.array([-cx[e],-cy[e],-cz[e],
                                   cx[e],cy[e],cz[e]]), U[np.ix_(elemDof)])
N = S*A
```

---

### Appendix B: Optimization Algorithms Source Code

#### B.1. Differential Evolution (DE)
Implementation of the standard DE algorithm.

```python
import numpy as np
import pandas as pd
from pyDOE import lhs

#define Differential Evolution algorithm
def DE(bounds, mut, crossp, popsize, its):
    numberFEA = 0
    pop = lhs(nVar,samples=popsize)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = [None]*popsize
    
    for j in range(popsize):
        cv = StructAnalysis(pop_denorm[j])
        numberFEA = numberFEA + 1
        fitness[j] = fitnessfunc(pop_denorm[j], cv)
        
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_denorm = pop_denorm[best_idx]
    
    for i in range(1, its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            x1, x2 = pop[np.random.choice(idxs, 2, replace = False)]
            mutant = np.clip(pop[j]+mut*(best-pop[j])+mut*(x1-x2),0,1)
            cross_points = np.random.rand(nVar) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, nVar)] = True
                
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            cv = StructAnalysis(trial_denorm)
            numberFEA = numberFEA + 1
            f = fitnessfunc(trial_denorm, cv)
            
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
            if f < fitness[best_idx]:
                best_idx = j
                best = trial
                best_denorm = trial_denorm
                
        yield best_denorm, \
              fitness[best_idx], \
              numberFEA
```

#### B.2. Classification-assisted Differential Evolution (CaDE)
Implementation of the proposed CaDE method integrating AdaBoost.

```python
import numpy as np
import pandas as pd
from pyDOE import lhs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#define function for assigning label 
def assign_label(cv):
    if cv > 0.0:
        return -1
    else:
        return +1

#define Classifier-assisted Differential Evolution
def CaDE(mut, crossp, popsize, n_iter1, its):
    numberFEA = 0
    #initial iteraction
    nVar = len(bounds)
    pop = lhs(nVar,samples=popsize)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = [None]*popsize
    x_train = pop_denorm
    y_train = np.empty(shape=(popsize,1))
    
    for j in range(popsize):
        cv = StructAnalysis(pop_denorm[j])
        numberFEA = numberFEA + 1
        fitness[j] = fitnessfunc(pop_denorm[j], cv)
        y_train[[j,0]] = assign_label(cv)
        
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_denorm = pop_denorm[best_idx]
    
    # phase I: model building phase
    for i in range(1, n_iter1):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            x1, x2 = pop[np.random.choice(idxs, 2, replace = False)]
            mutant = np.clip(pop[j]+mut*(best-pop[j])+mut*(x1-x2),0,1)
            cross_points = np.random.rand(nVar) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, nVar)] = True
            
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trial_cv = StructAnalysis(trial_denorm)
            numberFEA = numberFEA + 1
            trial_f = fitnessfunc(trial_denorm, trial_cv)
            
            if trial_f < fitness[j]:
                fitness[j] = trial_f
                pop[j] = trial
                pop_denorm[j] = trial_denorm
            if trial_f < fitness[best_idx]:
                best_idx = j
                best = trial
                best_denorm = trial_denorm
                
            #add more data points
            x_train = np.concatenate((x_train, \
                                      trial_denorm.reshape((1,nVar))))
            y_train = np.concatenate((y_train, \
                                      np.asarray([[trial_cv]])))
        yield best_denorm, \
              fitness[best_idx], \
              numberFEA
              
    #defining and training classification model
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             n_estimators=50)
    clf.fit(x_train, y_train.ravel())
    
    #phase II: model employing phase
    for i in range(n_iter1, its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            x1, x2 = pop[np.random.choice(idxs, 2, replace = False)]
            mutant = np.clip(pop[j]+mut*(best-pop[j])+mut*(x1-x2),0,1)
            cross_points = np.random.rand(nVar) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, nVar)] = True
                
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trial_label_pred=clf.predict(trial_denorm.reshape(1,nVar))
            
            if trial_label_pred == 1:
                trial_cv = StructAnalysis(trial_denorm)
                numberFEA = numberFEA + 1
                trial_f = fitnessfunc(trial_denorm, trial_cv)
                if trial_f < fitness[j]:
                    fitness[j] = trial_f
                    pop[j] = trial
                    pop_denorm[j] = trial_denorm
                if trial_f < fitness[best_idx]:
                    best_idx = j
                    best = trial
                    best_denorm = trial_denorm
                    
            elif trial_label_pred == -1:
                if weight(trial_denorm) < weight(pop_denorm[j]):
                    trial_cv = StructAnalysis(trial_denorm)
                    numberFEA = numberFEA + 1
                    trial_f = fitnessfunc(trial_denorm, trial_cv)
                    if trial_f < fitness[j]:
                        fitness[j] = trial_f
                        pop[j] = trial
                        pop_denorm[j] = trial_denorm
                    if trial_f < fitness[best_idx]:
                        best_idx = j
                        best = trial
                        best_denorm = trial_denorm
        
        yield best_denorm, \
              fitness[best_idx], \
              numberFEA

