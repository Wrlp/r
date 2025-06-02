#########################################
#               Partie 2                #
#########################################

import numpy as np
import matplotlib.pyplot as plt

# Données
points = [(1,1), (1,2), (1,5), (3,4), (4,3), (6,2), (0,4)]
x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])


# 2.1 Calcul des Coefficients de Régression

# Moyennes
x_moy = np.mean(x)
y_moy = np.mean(y)

# Coefficients
b1 = np.sum((x - x_moy) * (y - y_moy)) / np.sum((x - x_moy)**2)
b0 = y_moy - b1 * x_moy

# Modèle de prédictions
y_estime = b0 + b1 * x


# 2.3 Coefficient de Détermination R²

# R²
SCE = np.sum((y - y_estime) ** 2)
SCT = np.sum((y - y_moy) ** 2)
SCR = np.sum((y_estime - y_moy) ** 2)
R2 = 1 - SCE / SCT


# 2.2 Visualisation de la Droite de Régression

# Affichage
plt.scatter(x, y, label='Points')
plt.plot(x, y_estime, color='red', label=f'Droite de régression: y = {b0:.2f} + {b1:.2f}x')
plt.legend()
plt.title(f'Droite de régression linéaire simple (R² = {R2:.2f})')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
plt.savefig("figure_2_1.jpg")
print("Graphique enregistré dans le fichier figure_2_1.jpg")


#########################################
#               Partie 5                #
#########################################

# Classification Ascendante Hiérarchique (CAH)

import math
import pandas as pd

# 1.a
def dist(p1, p2):
    """Distance euclidienne"""
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    print("La distance euclidienne vaut : ", distance)
    return distance

# 1.b
def dist1(p1, p2):
    """Distance de Manhattan"""
    distance = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    print("La distance de Manhattan vaut : ", distance)
    return distance

# 1.c
def dist_inf(p1, p2):
    """Distance de Chebyshev (max)"""
    distance = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    print("La distance de Chebyshev vaut : ", distance)
    return distance

# 1.d : Distance de Ward ???


# 2.
def dist_min(tableau, dist_func):
    min_d = float('inf')
    couple_points = None
    for i in range(len(tableau)):
        for j in range(i+1, len(tableau)):
            d = dist_func(tableau[i], tableau[j])
            if d < min_d:
                min_d = d
                couple_points = (tableau[i], tableau[j])
    return couple_points, min_d


# 3.
points = {
    "M1": (1, 1),
    "M2": (1, 2),
    "M3": (1, 5),
    "M4": (3, 4),
    "M5": (4, 3),
    "M6": (6, 2),
    "M7": (0, 4)
}

# Tracé
plt.figure(figsize=(6, 6))
for name, (x, y) in points.items():
    plt.scatter(x, y, color='blue')
    plt.text(x + 0.1, y, name, fontsize=12)

# Initialisation de la matrice
n = len(points)
matrice_1 = np.zeros((n, n))

# Remplissage de la matrice avec d²
for i in range(n):
    for j in range(n):
        xi, yi = points[names[i]]
        xj, yj = points[names[j]]
        d_squared = (xi - xj)**2 + (yi - yj)**2
        matrice_1[i][j] = d_squared

# Affichage avec pandas pour lisibilité
df = pd.DataFrame(matrice_1, index=names, columns=names)
print("Matrice des distances euclidiennes au carré :\n")
print(df.round(1))

# Encadrer M3 et M7 (Classe Γ₁)
x_vals = [points["M1"][0], points["M7"][0]]
y_vals = [points["M1"][1], points["M7"][1]]
plt.plot(x_vals, y_vals, 'ro--')
plt.scatter(x_vals, y_vals, color='red')
plt.title("Regroupement initial : Classe Γ₁ = {M1, M7}")
plt.grid(True)
plt.xlim(-1, 7)
plt.ylim(0, 6)
plt.show()

plt.savefig("figure_5_3.jpg")
print("Graphique enregistré dans le fichier figure_5_3.jpg")