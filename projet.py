import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

points =[(1,1),(1,2),(1,5),(3,4),(4,3),(6,2),(0,4)]
noms= ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]


#########################################
#               Partie 1                #
#########################################

#Séparer les coordonnées x et y
x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])


def afficher_stats(valeurs, noms):
    print("Moyenne :", np.mean(valeurs))
    print("Médiane :", np.median(valeurs))
    print("Variance :", np.var(valeurs))
    print("Ecart-type :", np.std(valeurs))
    print("Minimum :", np.min(valeurs))
    print("Maximum :", np.max(valeurs))
    print("Etendue :", np.ptp(valeurs))


afficher_stats(x, "x")
afficher_stats(y, "y")

plt.scatter(x, y, color='blue')
for i in range(len(points)):
    plt.text(x[i] + 0.1, y[i] + 0.1, noms[i])
plt.title("Nuage de points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

#   Interprétation statistique :

#Les valeurs xx sont plus dispersées que les valeurs yy.
#La moyenne de xx est plus faible que celle de yy, indiquant une asymétrie spatiale.
#L’étendue plus grande pour xx traduit une plus grande variabilité sur l’axe horizontal.

#   Interprétation graphique :

#Le nuage montre une certaine tendance croissante mais non strictement linéaire.
#Il est possible qu’une relation linéaire approximative existe entre xx et yy, à tester dans la partie suivante.


#########################################
#               Partie 2                #
#########################################

# 2.1 Calcul des Coefficients de Régression

# Moyennes
x_moy = np.mean(x)
y_moy = np.mean(y)

# Coefficients
b1 = np.sum((x - x_moy) * (y - y_moy)) / np.sum((x - x_moy)**2)
b0 = y_moy - b1 * x_moy

print("bo =", b0)
print("b1 =", b1)

# Modèle de prédictions
y_pred = b0 + b1 * x


# 2.3 Coefficient de Détermination R²

# R²
SCE = np.sum((y - y_pred) ** 2)
SCT = np.sum((y - y_moy) ** 2)
SCR = np.sum((y_pred - y_moy) ** 2)
R2 = 1 - SCE / SCT
print("R² (Coefficient de détermination) :", R2)


# 2.2 Visualisation de la Droite de Régression

# Affichage
plt.scatter(x, y, label='Points')
plt.plot(x, y_pred, color='red', label=f'Droite de régression: y = {b0:.2f} + {b1:.2f}x')
plt.legend()
plt.title(f'Droite de régression linéaire simple (R² = {R2:.2f})')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
plt.savefig("figure_2_1.jpg")
print("Graphique enregistré dans le fichier figure_2_1.jpg")


#########################################
#               Partie 3                #
#########################################

#1. Résidus et somme des carrés des erreurs (SCE)
e = y - y_pred

SCE=0

for i in range(len(points)):
    SCE += e[i]**2 
print("SCE (Somme des carrés des erreurs) :", SCE)

#2. Etimation de la variance des erreurs : MSE
n= len(x)

MSE = SCE / (n-2)

print("MSE (Erreur quadratique moyenne) :", MSE)

#Le n-2 vient du fait qu'on a estimé 2 paramètres (b₀ et b₁). On parle alors de degrés de liberté.

#3. Ecart-type des erreurs
s = np.sqrt(MSE)

print("Écart-type des erreurs :", s)

#4. Interprétation des Résultats


#########################################
#               Partie 4                #
#########################################

# 4.1 Estimation de la variance des erreurs

# 4.2 Erreurs Standards des Coefficients

# Erreur standard de la pente
somme_x=0
for i in range(len(points)):
    somme_x += (x[i]-x_moy)**2

SEb1 = s/np.sqrt(somme_x)
print("Erreur standard de la pente:", SEb1)

# Erreur standard de l'ordonnée à l'origine
SEb0 = s*np.sqrt(1/n+x_moy**2/somme_x)
print("Erreur standard de l'ordonnée à l'origine", SEb0)

# 4.3 Test d'Hypothèse pour la Pente (b1)

# Hypothèses :
# H0 : b1 = 0 (pas de relation linéaire)
# H1 : b1 != 0 (relation linéaire significative)

# Statistique de test

t_b1 = (b1-0)/SEb1 # t suit une loi de Student donc on calcul la p-valeur pour savoir si on rejette l'hypothèse ou non
p_b1 = 2 * (1 - stats.t.cdf(abs(t_b1), df=n - 2))
print("valeur de test", t_b1)
print("p-valeur:", p_b1)
if(p_b1<0.05):
    print("p-valeur<0,05 donc on rejette l'hypothèse, b1 est significatif")
else:
    print("p-valeur>0,05 donc on ne rejette pas l'hypothèse, il n'y a pas de preuve de relation linéaire")
    

# 4.4 Test d'Hypothèse pour l'Ordonnée à l'Origine (b0)
# Hypothèses : H0 : b0 = 0 vs H1 : b0 != 0

# Statistique de test

t_b0 = (b0-0)/SEb0
p_b0 = 2 * (1 - stats.t.cdf(abs(t_b0), df=n - 2))
print("valeur de test", t_b0)
print("p-valeur:", p_b0)
if(p_b0<0.05):
    print("p-valeur<0,05 donc on rejette l'hypothèse, b0 est significatif")
else:
    print("p-valeur>0,05 donc on ne rejette pas l'hypothèse, il n'y a pas de preuve de relation linéaire")

# 4.5 Intervalles de Confiance pour les Coefficients

alpha = 0.05

t_crit = stats.t.cdf(1-alpha/2, df= n-2)

IC_b1=[float(b1-t_crit*SEb1),float(b1+t_crit*SEb1)]
IC_b0=[float(b0-t_crit*SEb0),float(b0+t_crit*SEb0)]

print("Intervalle de confiance à 95% pour b1:", IC_b1)
print("Intervalle de confiance à 95% pour b0:", IC_b0)

# 4.6 Interprétation des Tests Statistiques

