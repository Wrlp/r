import numpy as np

points =[(1,1),(1,2),(1,5),(3,4),(4,3),(6,2),(0,4)]

def afficher_stats(valeurs):
    print("Moyenne :", np.mean(valeurs))
    print("Médiane :", np.median(valeurs))
    print("Variance :", np.var(valeurs))
    prnt("")
