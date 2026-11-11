"""
Lee las O_tilde guardadas por correr_anillo_par.py y construye la matriz de covarianza
del estado fundamental en la base de los sitios:

    M = O_tilde^T  M'  O_tilde,        M' = [[0, -1], [1, 0]]   (vacío de los modos)

Orden de los Majoranas: (a_1..a_N | b_1..b_N),  a_i = c_i + c_i^†,  b_i = i (c_i - c_i^†).
M_{alpha beta} = <-i gamma_alpha gamma_beta>  (alpha != beta).

Uso: poner este archivo en la misma carpeta que los *_O_tilde.txt, fijar gamma, h y N
en el main y ejecutar. Guarda <nombre>_M.txt en esa carpeta.

Requiere: mpmath   (pip install mpmath)
"""
import os
import time
from mpmath import mp

DPS = 50


def leer_matriz(archivo):
    """Lee una matriz guardada como texto (una fila por línea)."""
    with open(archivo) as f:
        filas = [[mp.mpf(x) for x in linea.split()] for linea in f if linea.strip()]
    return mp.matrix(filas)


def M_prima(n2):
    """Covarianza trivial del vacío de los modos, orden por bloques: [[0, -1], [1, 0]]."""
    N = n2 // 2
    Mp = mp.zeros(n2, n2)
    for k in range(N):
        Mp[k, N + k] = -1
        Mp[N + k, k] = 1
    return Mp


def covarianza(Ot):
    """M = O_tilde^T M' O_tilde."""
    return Ot.T * (M_prima(Ot.rows) * Ot)


def guardar_matriz(archivo, A):
    with open(archivo, "w") as f:
        for i in range(A.rows):
            f.write(" ".join(mp.nstr(A[i, j], DPS) for j in range(A.cols)) + "\n")


def controles(M):
    """Antisimetría, bloques aa y bb nulos, y pureza M^2 = -1 (en 5 filas)."""
    n2 = M.rows
    N = n2 // 2
    antisim = max(abs(M[i, j] + M[j, i]) for i in range(n2) for j in range(n2))
    aa_bb = max(max(abs(M[i, j]) for i in range(N) for j in range(N)),
                max(abs(M[N + i, N + j]) for i in range(N) for j in range(N)))
    pureza = mp.mpf(0)
    for i in range(0, n2, max(1, n2 // 5)):
        for j in range(n2):
            fila = mp.fsum(M[i, k] * M[k, j] for k in range(n2))
            pureza = max(pureza, abs(fila + (1 if i == j else 0)))
    return antisim, aa_bb, pureza


def covarianza_caso(gamma, h, N):
    """Lee la O_tilde de (gamma, h, N) de Resultados_Ham_Solver,
    calcula M y la guarda en Resultados_covariance_matrix."""
    mp.dps = DPS
    programas = os.path.dirname(os.path.abspath(__file__))
    entrada = os.path.join(programas, "Resultados_Ham_Solver")
    salida = os.path.join(programas, "Resultados_covariance_matrix")
    os.makedirs(salida, exist_ok=True)

    nombre = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}"
    archivo = os.path.join(entrada, nombre + "_O_tilde.txt")
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No encontré {nombre}_O_tilde.txt en {entrada}")

    t0 = time.time()
    Ot = leer_matriz(archivo)
    M = covarianza(Ot)
    guardar_matriz(os.path.join(salida, nombre + "_M.txt"), M)
    antisim, aa_bb, pureza = controles(M)
    print(f"{nombre}:  |M + M^T| = {mp.nstr(antisim, 3)},  |bloques aa, bb| = {mp.nstr(aa_bb, 3)},  "
          f"|M^2 + 1| = {mp.nstr(pureza, 3)}   -> Resultados_covariance_matrix/{nombre}_M.txt  "
          f"({time.time() - t0:.0f} s)", flush=True)
    return M

if __name__ == "__main__":
    mp.dps = DPS

    gamma = 1.0
    h = 0.5
    N = 200

    M = covarianza_caso(gamma, h, N)

"""
if __name__ == "__main__":
    import glob
    import re

    mp.dps = DPS
    programas = os.path.dirname(os.path.abspath(__file__))
    entrada = os.path.join(programas, "Resultados_Ham_Solver")
    patron = re.compile(r"anillo_par_N(\d+)_gamma([-\d.]+)_h([-\d.]+)_O_tilde\.txt$")

    for archivo in sorted(glob.glob(os.path.join(entrada, "*_O_tilde.txt"))):
        m = patron.search(os.path.basename(archivo))
        if m is None:
            continue
        N, gamma, h = int(m.group(1)), float(m.group(2)), float(m.group(3))
        covarianza_caso(gamma, h, N)
"""