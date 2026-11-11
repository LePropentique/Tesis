"""
Diagonalización local de la matriz de covarianza del anillo en una partición en arcos:
tripartición A | B | C  o  bipartición S | T.

Lee M de Resultados_covariance_matrix/dps<n> (orden (a_1..a_N | b_1..b_N)), y para cada región R:
    M_RR = [[0, W_RR], [-W_RR^T, 0]]    (orden (a_R | b_R))
    W_RR = U diag(lam) V^T   (SVD, precisión arbitraria)
    O_R  = [[U^T, 0], [0, -V^T]]   ->   O_R M_RR O_R^T = [[0, -diag(lam)], [diag(lam), 0]]
Con O = suma directa de las O_R calcula  M_tilde = O M O^T  y la guarda en
Resultados_diag_local/dps<n>.

Orden de M_tilde: (a'_R1 | b'_R1 | a'_R2 | b'_R2 | ...), con los modos de cada región
ordenados de menor a mayor lam (del más entrelazado al menos).

Regiones: arcos del anillo, con los sitios numerados 0..N-1. Un arco puede cruzar el enlace
(N-1, 0); sus sitios se recorren en el orden del anillo (p. ej. C seguido de A).

    Tripartición:  A = [0, NA),  B = [NA, NA+NB),  C = [NA+NB, N)
                   archivos <base>_A{NA}_B{NB}_C{NC}_...   con etiquetas A, B, C
    Bipartición:   S = arco de NS sitios que empieza en 'inicio',  T = el resto del anillo,
                   recorrido desde donde termina S
                   archivos <base>_bip_ini{inicio}_NS{NS}_...  con etiquetas S, T
    Biparticiones asociadas a A | B | C (siempre las tres):
                   A|BC:  S = A,   T = B, C
                   B|CA:  S = B,   T = C, A
                   AB|C:  S = A, B,  T = C

Archivos de cada partición (con <prefijo> como arriba):
    <prefijo>_Mtilde.txt
    <prefijo>_O_<R>.txt        ortogonal local de la región R (orden (a_R | b_R))
    <prefijo>_lambda_<R>.txt   lambdas de la región R (de menor a mayor)

Requiere: mpmath   (pip install mpmath)
"""
import os
import time
from mpmath import mp

DPS = 70
PROGRAMAS = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Lectura y escritura
# ---------------------------------------------------------------------------
def leer_matriz(archivo):
    with open(archivo) as f:
        filas = [[mp.mpf(x) for x in linea.split()] for linea in f if linea.strip()]
    return mp.matrix(filas)


def guardar_matriz(archivo, A):
    with open(archivo, "w") as f:
        for i in range(A.rows):
            f.write(" ".join(mp.nstr(A[i, j], DPS) for j in range(A.cols)) + "\n")


def submatriz(M, filas, cols):
    S = mp.matrix(len(filas), len(cols))
    for p, i in enumerate(filas):
        for q, j in enumerate(cols):
            S[p, q] = M[i, j]
    return S


# ---------------------------------------------------------------------------
# Particiones en arcos y nombres de archivo
# ---------------------------------------------------------------------------
def arco(inicio, longitud, N):
    """Sitios (0..N-1) de un arco de 'longitud' sitios que empieza en 'inicio'."""
    return [(inicio + i) % N for i in range(longitud)]


def base_caso(gamma, h, N):
    return f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}"


def triparticion(N, NA, NB):
    """Regiones, etiquetas y sufijo de nombre de la tripartición A | B | C."""
    NC = N - NA - NB
    if min(NA, NB, NC) < 1:
        raise ValueError(f"Cada región debe tener al menos un sitio (A|B|C = {NA}|{NB}|{NC}).")
    regiones = [arco(0, NA, N), arco(NA, NB, N), arco(NA + NB, NC, N)]
    return regiones, "ABC", f"A{NA}_B{NB}_C{NC}"


def biparticion(N, inicio, NS):
    """Regiones, etiquetas y sufijo de nombre de la bipartición S | T."""
    if not 1 <= NS <= N - 1:
        raise ValueError(f"S debe tener entre 1 y N-1 sitios (NS = {NS}, N = {N}).")
    inicio %= N
    regiones = [arco(inicio, NS, N), arco(inicio + NS, N - NS, N)]
    return regiones, "ST", f"bip_ini{inicio}_NS{NS}"


def biparticiones_asociadas(N, NA, NB):
    """Las tres biparticiones de A | B | C, como (nombre, inicio de S, NS)."""
    return [("A|BC", 0, NA), ("B|CA", NA, NB), ("AB|C", 0, NA + NB)]


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
def ortogonal_local(M, region, N):
    """O_R y lam de una región, a partir de la SVD de W_RR (W = bloque a-b de M)."""
    R = list(region)
    W_RR = submatriz(M, R, [N + i for i in R])
    U, S, Vt = mp.svd_r(W_RR)                     # W_RR = U diag(S) Vt
    orden = sorted(range(len(R)), key=lambda m: S[m])
    n = len(R)
    O_R = mp.zeros(2 * n, 2 * n)
    for fila, m in enumerate(orden):
        for x in range(n):
            O_R[fila, x] = U[x, m]                # a'_m = sum_x U[x, m] a_x
            O_R[n + fila, n + x] = -Vt[m, x]      # b'_m = -sum_x V[x, m] b_x
    lam = [S[m] for m in orden]
    return O_R, lam


def diagonalizacion_local(M, regiones):
    """M_tilde = O M_perm O^T, calculado bloque por bloque:  M_tilde_XY = O_X M_XY O_Y^T.
    'regiones' es una lista de listas de sitios (cualquier número de regiones)."""
    N = M.rows // 2
    idx = [list(R) + [N + i for i in R] for R in regiones]   # índices (a_R | b_R) en M
    O = []
    lams = []
    for R in regiones:
        O_R, lam = ortogonal_local(M, R, N)
        O.append(O_R)
        lams.append(lam)

    n2 = M.rows
    M_tilde = mp.matrix(n2, n2)
    inicio = [0]
    for I in idx:
        inicio.append(inicio[-1] + len(I))
    for X, IX in enumerate(idx):
        for Y, IY in enumerate(idx):
            bloque = O[X] * submatriz(M, IX, IY) * O[Y].T
            for p in range(bloque.rows):
                for q in range(bloque.cols):
                    M_tilde[inicio[X] + p, inicio[Y] + q] = bloque[p, q]
    return M_tilde, O, lams


def error_canonico(M_tilde, lams):
    """max |bloques diagonales de M_tilde - [[0, -lam], [lam, 0]]|."""
    err = mp.mpf(0)
    ini = 0
    for lam in lams:
        n = len(lam)
        for p in range(2 * n):
            for q in range(2 * n):
                esperado = 0
                if q == p + n:
                    esperado = -lam[p]
                elif p == q + n:
                    esperado = lam[q]
                err = max(err, abs(M_tilde[ini + p, ini + q] - esperado))
        ini += 2 * n
    return err


# ---------------------------------------------------------------------------
# Casos: leen M, calculan, guardan (o reutilizan si ya existe)
# ---------------------------------------------------------------------------
def _carpeta(nombre, dps):
    return os.path.join(PROGRAMAS, nombre, f"dps{dps}")


def ya_calculado(prefijo, etiquetas, dps):
    carpeta = _carpeta("Resultados_diag_local", dps)
    archivos = [f"{prefijo}_Mtilde.txt"] + [f"{prefijo}_{t}_{r}.txt"
                                            for r in etiquetas for t in ("O", "lambda")]
    return all(os.path.exists(os.path.join(carpeta, a)) for a in archivos)


def leer_resultado(prefijo, etiquetas, dps):
    """(M_tilde, [O_R], [lam_R]) de una partición ya guardada."""
    carpeta = _carpeta("Resultados_diag_local", dps)
    M_tilde = leer_matriz(os.path.join(carpeta, f"{prefijo}_Mtilde.txt"))
    O = [leer_matriz(os.path.join(carpeta, f"{prefijo}_O_{r}.txt")) for r in etiquetas]
    lams = []
    for r in etiquetas:
        with open(os.path.join(carpeta, f"{prefijo}_lambda_{r}.txt")) as f:
            lams.append([mp.mpf(x) for x in f.read().split()])
    return M_tilde, O, lams


def guardar_resultado(prefijo, etiquetas, M_tilde, O, lams, dps):
    carpeta = _carpeta("Resultados_diag_local", dps)
    os.makedirs(carpeta, exist_ok=True)
    guardar_matriz(os.path.join(carpeta, f"{prefijo}_Mtilde.txt"), M_tilde)
    for r, O_R, lam in zip(etiquetas, O, lams):
        guardar_matriz(os.path.join(carpeta, f"{prefijo}_O_{r}.txt"), O_R)
        with open(os.path.join(carpeta, f"{prefijo}_lambda_{r}.txt"), "w") as f:
            for l in lam:
                f.write(mp.nstr(l, dps) + "\n")


def leer_covarianza(gamma, h, N, dps):
    base = base_caso(gamma, h, N)
    archivo = os.path.join(_carpeta("Resultados_covariance_matrix", dps), base + "_M.txt")
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No encontré {base}_M.txt en Resultados_covariance_matrix/dps{dps}")
    return leer_matriz(archivo)


def diag_particion(gamma, h, N, regiones, etiquetas, sufijo, dps=None, M=None, log=print):
    """Diagonalización local de una partición cualquiera en arcos. Si ya está guardada la lee;
    si no, la calcula (leyendo M si no se la pasan) y la guarda.
    Devuelve (M_tilde, [O_R], [lam_R], prefijo)."""
    global DPS
    if dps is not None:
        DPS = dps
    mp.dps = DPS
    prefijo = f"{base_caso(gamma, h, N)}_{sufijo}"
    if ya_calculado(prefijo, etiquetas, DPS):
        log(f"{prefijo}: ya calculado en dps{DPS}, lo leo.")
        return (*leer_resultado(prefijo, etiquetas, DPS), prefijo)

    t0 = time.time()
    if M is None:
        M = leer_covarianza(gamma, h, N, DPS)
    M_tilde, O, lams = diagonalizacion_local(M, regiones)
    guardar_resultado(prefijo, etiquetas, M_tilde, O, lams, DPS)
    log(f"{prefijo}:  lam_min " + ", ".join(etiquetas) + " = "
        + ", ".join(mp.nstr(l[0], 6) for l in lams)
        + f",  |bloques diagonales - forma canónica| = {mp.nstr(error_canonico(M_tilde, lams), 3)}"
        + f"  ({time.time() - t0:.0f} s)")
    return M_tilde, O, lams, prefijo


def diag_local_caso(gamma, h, N, NA, NB, dps=None, log=print):
    """Tripartición A | B | C (C es el resto)."""
    regiones, etiquetas, sufijo = triparticion(N, NA, NB)
    return diag_particion(gamma, h, N, regiones, etiquetas, sufijo, dps, log=log)


def diag_biparticion_caso(gamma, h, N, NS, inicio=0, dps=None, M=None, log=print):
    """Bipartición S | T, con S = arco de NS sitios desde 'inicio' y T el resto."""
    regiones, etiquetas, sufijo = biparticion(N, inicio, NS)
    return diag_particion(gamma, h, N, regiones, etiquetas, sufijo, dps, M=M, log=log)


def diag_asociadas_caso(gamma, h, N, NA, NB, dps=None, log=print):
    """Las tres biparticiones de A | B | C: A|BC, B|CA, AB|C.
    Devuelve un diccionario  nombre -> (M_tilde, [O_S, O_T], [lam_S, lam_T], prefijo)."""
    triparticion(N, NA, NB)                          # solo para validar las regiones
    if dps is not None:
        mp.dps = dps
    M = None
    resultados = {}
    for nombre, inicio, NS in biparticiones_asociadas(N, NA, NB):
        _, etiquetas, sufijo = biparticion(N, inicio, NS)
        prefijo = f"{base_caso(gamma, h, N)}_{sufijo}"
        if M is None and not ya_calculado(prefijo, etiquetas, dps or DPS):
            M = leer_covarianza(gamma, h, N, dps or DPS)   # se lee una sola vez para las tres
        log(f"[{nombre}]")
        resultados[nombre] = diag_biparticion_caso(gamma, h, N, NS, inicio, dps, M=M, log=log)
    return resultados


if __name__ == "__main__":
    gamma = 0.5
    h = 0.3
    N = 200
    NA, NB = 66, 66          # C es el resto: N - NA - NB
    dps = 70

    diag_local_caso(gamma, h, N, NA, NB, dps)
    diag_asociadas_caso(gamma, h, N, NA, NB, dps)