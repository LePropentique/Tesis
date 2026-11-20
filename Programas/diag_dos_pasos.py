"""
Diagonalización de los bloques cruzados en DOS PASOS (más un paso de orden), para la
tripartición A | B | C con regiones IGUALES (NA = NB = NC = n) en el anillo.

Por la simetría de traslación, los bloques cruzados de W (bloque a-b de la covarianza, en la
base de los sitios) forman dos familias de matrices iguales salvo signo:
    familia AB:  W_AB = W_BC = -W_CA          familia BA:  W_BA = W_CB = -W_AC
Por eso se usa LA MISMA rotación en las tres regiones: U para los a y V para los b,
    A = diag(U^T, U^T, U^T),   C = diag(V, V, V),     (A W C)_XY = U^T W_XY V.
Los nuevos Majoranas son a'_p = sum_k U[k, p] a_k  y  b'_p = sum_k V[k, p] b_k (columnas de U, V).

Pasos (todo en precisión arbitraria, con mpmath):
  1. Familia AB:  SVD  W_AB = U S V^T.  AB, BC y CA quedan diagonales.
  2. Familia BA, en la "zona libre":  en las posiciones p..n-1, donde los acoples de AB son
     prácticamente cero, se hace la SVD del bloque  [U^T W_BA V]_{p:, p:} = P S' Q^T  y
         U[:, p:] -> U[:, p:] P,      V[:, p:] -> V[:, p:] Q.
     Las posiciones 0..p-1 no se tocan, así que AB sigue diagonal; en la zona libre BA queda
     diagonal. Funciona porque las dos familias ocupan subespacios casi ortogonales.
  3. Orden de los modos que no se acoplan con ninguna familia (lambda ~ 1): en las posiciones
     cuyo acople cruzado es despreciable (ver tol_cola), el bloque local [U^T W_AA V] es una
     ortogonal con todos sus valores singulares ~1 y su base es arbitraria; una SVD de ese
     bloque (Williamson) la ordena. No cambia los bloques cruzados más que en tol_cola.

p (número de posiciones reservadas a AB) se elige, si no se da, probando todos los valores en
doble precisión y tomando el de menor j (peso fuera de la diagonal de los 6 bloques cruzados).

Orden de las posiciones en el resultado: el de la construcción (primero las p de AB, luego las
de BA de mayor a menor acople, al final los modos sin acoples).

Archivos (en Resultados_diag_dos_pasos/dps<n>, con <prefijo> como en diag_local):
    <prefijo>_dp_Mtilde.txt     M_tilde = O M O^T, orden (a'_A | b'_A | a'_B | b'_B | a'_C | b'_C)
    <prefijo>_dp_O_<R>.txt      ortogonal local de la región R, [[U^T, 0], [0, V^T]] (orden a_R | b_R)
    <prefijo>_dp_info.txt       p, j total y por bloque, sigma de cada familia, tiempo

Requiere: numpy, mpmath   (pip install numpy mpmath), y diag_local.py en la misma carpeta.
"""
import os
import time
import numpy as np
from mpmath import mp

DPS = 70
PROGRAMAS = os.path.dirname(os.path.abspath(__file__))
BLOQUES = ("AB", "BC", "CA", "BA", "CB", "AC")      # los seis bloques cruzados


# ---------------------------------------------------------------------------
# Álgebra en precisión arbitraria con listas de filas
# ---------------------------------------------------------------------------
def _producto(A, B):
    columnas = list(zip(*B))
    return [[mp.fdot(fila, col) for col in columnas] for fila in A]


def _transponer(T):
    return [list(c) for c in zip(*T)]


def _svd(T):
    """SVD de T (lista de filas, cuadrada): (U, s, V) con T = U diag(s) V^T, s de mayor a menor;
    U y V como listas de filas, con los vectores singulares en las COLUMNAS."""
    n = len(T)
    if n == 0:
        return [], [], []
    U, S, Vt = mp.svd_r(mp.matrix(T), full_matrices=True)
    orden = sorted(range(n), key=lambda i: -S[i])
    U = [[U[k, i] for i in orden] for k in range(n)]
    V = [[Vt[i, k] for i in orden] for k in range(n)]
    return U, [S[i] for i in orden], V


def _rotar_columnas(T, desde, R):
    """T[:, desde:] -> T[:, desde:] R  (R cuadrada, de tamaño n - desde)."""
    if not R:
        return
    columnas_R = [list(c) for c in zip(*R)]
    for fila in T:
        viejas = fila[desde:]
        fila[desde:] = [mp.fdot(viejas, col) for col in columnas_R]


def _bloques(W, n):
    """Bloques W_XY (listas de filas) de W (N x N, lista de filas), regiones de n sitios."""
    return {X + Y: [fila[n * j:n * j + n] for fila in W[n * i:n * i + n]]
            for i, X in enumerate("ABC") for j, Y in enumerate("ABC")}


def _transformar(Wb, U, V):
    """(A W C)_XY = U^T W_XY V para los nueve bloques."""
    Ut = _transponer(U)
    return {b: _producto(_producto(Ut, X), V) for b, X in Wb.items()}


def _fuera(T):
    """(peso fuera de la diagonal, peso total) de un bloque."""
    total = mp.fsum(x * x for fila in T for x in fila)
    diag = mp.fsum(T[i][i] ** 2 for i in range(len(T)))
    return total - diag, total


def medir(Tb):
    """j total (los seis bloques cruzados) y j de cada bloque."""
    por, F, T = {}, mp.mpf(0), mp.mpf(0)
    for b in BLOQUES:
        f, t = _fuera(Tb[b])
        por[b] = f / t
        F += f
        T += t
    return F / T, por


# ---------------------------------------------------------------------------
# Elección de p en doble precisión
# ---------------------------------------------------------------------------
def elegir_p(Wb):
    """El p (0..n) de menor j con los pasos 1 y 2, calculado en doble precisión.
    Devuelve (p, lista de j para cada p)."""
    W = {b: np.array([[float(x) for x in fila] for fila in X]) for b, X in Wb.items()}
    n = W["AB"].shape[0]
    U, _, Vt = np.linalg.svd(W["AB"])
    V = Vt.T
    js = []
    for p in range(n + 1):
        U2, V2 = U.copy(), V.copy()
        if p < n:
            P, _, Qt = np.linalg.svd((U.T @ W["BA"] @ V)[p:, p:])
            U2[:, p:] = U[:, p:] @ P
            V2[:, p:] = V[:, p:] @ Qt.T
        F = T = 0.0
        for b in BLOQUES:
            X = U2.T @ W[b] @ V2
            t = (X ** 2).sum()
            F += t - (np.diag(X) ** 2).sum()
            T += t
        js.append(F / T)
    return int(np.argmin(js)), js


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def diag_dos_pasos(M, n, p=None, tol_cola=1e-10, log=print):
    """
    M: covarianza (mp.matrix, 2N x 2N, orden (a_1..a_N | b_1..b_N)), con N = 3n.
    n: sitios por región (A = sitios 1..n, B = n+1..2n, C = 2n+1..3n).
    p: posiciones reservadas a la familia AB (None: se elige, ver elegir_p).
    tol_cola: una posición se considera sin acoples cruzados (paso 3) si su peso diagonal
        cruzado, sum_Y W''_XY[q, q]^2 con X = A, es menor que tol_cola (el peso de una fila es 1).
    Devuelve (M_tilde, O, info), en el formato de diag_local / diag_off.
    """
    t0 = time.time()
    N = M.rows // 2
    if N != 3 * n:
        raise ValueError(f"diag_dos_pasos necesita regiones iguales: N = {N} no es 3 x {n}.")
    W = [[M[i, N + j] for j in range(N)] for i in range(N)]
    Wb = _bloques(W, n)

    # p
    if p is None:
        p, js = elegir_p(Wb)
        log(f"p elegido = {p} (j en doble precisión = {js[p]:.2e})")
    p = max(0, min(int(p), n))

    # Paso 1: familia AB
    U, sAB, V = _svd(Wb["AB"])
    # Paso 2: familia BA en las posiciones p..n-1
    sBA = []
    if p < n:
        BA = _producto(_producto(_transponer(U), Wb["BA"]), V)
        P, sBA, Q = _svd([fila[p:] for fila in BA[p:]])
        _rotar_columnas(U, p, P)
        _rotar_columnas(V, p, Q)
    Tb = _transformar(Wb, U, V)
    j2, _ = medir(Tb)
    log(f"pasos 1 y 2: j = {mp.nstr(j2, 3)}")

    # Paso 3: Williamson en las posiciones sin acoples cruzados (las del final)
    peso = [mp.fsum(Tb[b][q][q] ** 2 for b in ("AB", "AC")) for q in range(n)]
    q0 = n
    while q0 > 0 and peso[q0 - 1] < tol_cola:
        q0 -= 1
    if q0 < n - 1:
        P3, _, Q3 = _svd([fila[q0:] for fila in Tb["AA"][q0:]])
        _rotar_columnas(U, q0, P3)
        _rotar_columnas(V, q0, Q3)
        Tb = _transformar(Wb, U, V)
    j, por = medir(Tb)
    log(f"paso 3 (posiciones {q0}..{n - 1}): j = {mp.nstr(j, 3)}   ({time.time() - t0:.0f} s)")

    info = dict(metodo="SVD en dos pasos, misma rotación en las tres regiones", p=p, q0=q0,
                j_dos_pasos=float(j2), j_final=float(j), por_bloque={b: float(x) for b, x in por.items()},
                sigma_AB=[float(x) for x in sAB], sigma_BA=[float(x) for x in sBA],
                segundos=time.time() - t0)

    # M_tilde (orden a'_A | b'_A | a'_B | b'_B | a'_C | b'_C) y ortogonales locales
    M_tilde = mp.zeros(2 * N, 2 * N)
    for i, X in enumerate("ABC"):
        for k, Y in enumerate("ABC"):
            T = Tb[X + Y]
            for a in range(n):
                for b in range(n):
                    M_tilde[2 * n * i + a, 2 * n * k + n + b] = T[a][b]
                    M_tilde[2 * n * k + n + b, 2 * n * i + a] = -T[a][b]
    O_R = mp.zeros(2 * n, 2 * n)
    for a in range(n):
        for k in range(n):
            O_R[a, k] = U[k][a]                 # filas = nuevos a' (U^T)
            O_R[n + a, n + k] = V[k][a]         # filas = nuevos b' (V^T)
    return M_tilde, [O_R, O_R.copy(), O_R.copy()], info


# ---------------------------------------------------------------------------
# Casos: leen M, calculan, guardan (o reutilizan si ya existe)
# ---------------------------------------------------------------------------
def _carpeta(dps):
    return os.path.join(PROGRAMAS, "Resultados_diag_dos_pasos", f"dps{dps}")


def ya_calculado(prefijo, dps):
    archivos = [f"{prefijo}_dp_Mtilde.txt", f"{prefijo}_dp_info.txt"] + [
        f"{prefijo}_dp_O_{r}.txt" for r in "ABC"]
    return all(os.path.exists(os.path.join(_carpeta(dps), a)) for a in archivos)


def leer_info(prefijo, dps):
    """El archivo de información de un caso guardado, como diccionario de textos."""
    info = {}
    with open(os.path.join(_carpeta(dps), f"{prefijo}_dp_info.txt"), encoding="utf-8") as f:
        for linea in f:
            if ":" in linea:
                clave, valor = linea.split(":", 1)
                info[clave.strip()] = valor.strip()
    return info


def leer_Mtilde(prefijo, dps):
    """M_tilde de un caso guardado, en doble precisión (suficiente para graficar)."""
    archivo = os.path.join(_carpeta(dps), f"{prefijo}_dp_Mtilde.txt")
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No encontré {prefijo}_dp_Mtilde.txt en {_carpeta(dps)}")
    return np.loadtxt(archivo)


def guardar_resultado(prefijo, M_tilde, O, info, dps):
    import diag_local
    diag_local.DPS = dps
    os.makedirs(_carpeta(dps), exist_ok=True)
    diag_local.guardar_matriz(os.path.join(_carpeta(dps), f"{prefijo}_dp_Mtilde.txt"), M_tilde)
    for r, O_R in zip("ABC", O):
        diag_local.guardar_matriz(os.path.join(_carpeta(dps), f"{prefijo}_dp_O_{r}.txt"), O_R)
    with open(os.path.join(_carpeta(dps), f"{prefijo}_dp_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"metodo: {info['metodo']}\n")
        f.write(f"p: {info['p']}\n")
        f.write(f"cola_desde: {info['q0']}\n")
        f.write(f"j_dos_pasos: {info['j_dos_pasos']:.6e}\n")
        f.write(f"j_final: {info['j_final']:.6e}\n")
        for b in BLOQUES:
            f.write(f"j_{b}: {info['por_bloque'][b]:.6e}\n")
        f.write(f"segundos: {info['segundos']:.0f}\n")
        f.write("sigma_AB: " + " ".join(f"{x:.6e}" for x in info["sigma_AB"]) + "\n")
        f.write("sigma_BA: " + " ".join(f"{x:.6e}" for x in info["sigma_BA"]) + "\n")


def dos_pasos_caso(gamma, h, N, dps=None, p=None, M=None, log=print):
    """Tripartición A | B | C con regiones iguales (N = 3n). Si ya está guardada no la repite;
    si no, la calcula (leyendo M si no se la pasan) y la guarda. Devuelve (prefijo, info)."""
    import diag_local
    global DPS
    if dps is not None:
        DPS = dps
    mp.dps = DPS
    if N % 3:
        raise ValueError(f"diag_dos_pasos necesita regiones iguales: N = {N} no es múltiplo de 3.")
    n = N // 3
    prefijo = f"{diag_local.base_caso(gamma, h, N)}_A{n}_B{n}_C{n}"
    if ya_calculado(prefijo, DPS):
        log(f"{prefijo}: dos pasos ya calculado en dps{DPS}.")
        return prefijo, leer_info(prefijo, DPS)
    if M is None:
        M = diag_local.leer_covarianza(gamma, h, N, DPS)
    M_tilde, O, info = diag_dos_pasos(M, n, p, log=log)
    guardar_resultado(prefijo, M_tilde, O, info, DPS)
    return prefijo, leer_info(prefijo, DPS)


if __name__ == "__main__":
    gamma = 0.5
    h = 0.3
    N = 198                  # regiones iguales: A | B | C = 66 | 66 | 66
    dps = 70

    dos_pasos_caso(gamma, h, N, dps)