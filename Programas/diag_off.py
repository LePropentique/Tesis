"""
Diagonalización simultánea aproximada de los bloques cruzados de la covarianza ("off-diag").

Dadas unas regiones R1, R2, ... (arcos del anillo), busca ortogonales locales que dejen lo más
diagonales posible, todos a la vez, los bloques de M entre regiones distintas:

    O_X M_XY O_Y^T  ~  diagonal        para cada par X != Y

(con tres regiones son las tres ecuaciones BA, CA, BC; con dos, una sola; etc.).

Como M es quiral (bloques aa y bb nulos), todo está en W = bloque a-b de M, y cada ortogonal
local es O_R = [[A_R, 0], [0, B_R]]: A_R mezcla los a de la región y B_R los b. Los nuevos
Majoranas son las FILAS de A_R y B_R (igual que en diag_local):

    W_tilde_XY = A_X W_XY B_Y^T ,        W_tilde[mu, nu] = <-i a'_mu b'_nu>.

"Diagonal" quiere decir que a'_{X,i} solo se acopla con b'_{Y,i}: la posición i es común a
todas las regiones. Lo que se minimiza es

    J = sum_{X != Y} || fuera de la diagonal de W_tilde_XY ||^2 ,     j = J / sum_{X != Y} ||W_XY||^2

(la norma de cada bloque no cambia, así que es lo mismo que maximizar el peso en la diagonal).

Cómo se resuelve
    Dos regiones: la solución es exacta, una SVD por bloque (W_XY y W_YX), en precisión arbitraria.
    Tres o más:
      1. Se parte de la base de Williamson de cada región (diag_local).
      2. Procrustes alternado (ver procrustes_alternado): con las demás quietas, cada A_X o B_Y
         se cambia por la ortogonal más cercana a G = sum (direcciones deseadas x acople actual).
         Se hace en doble precisión, que es rápido, y por ETAPAS: la doble precisión solo
         distingue acoples dentro de unas pocas cifras, así que primero se resuelven los más
         fuertes, se aplican en precisión arbitraria, y se repite con lo que queda, reescalado.
         En cada etapa se prueban varios arranques (la base actual y la SVD de cada par de
         regiones) y se toma el mejor, porque el método puede quedarse en un máximo local.
      3. Pulido: rotaciones de a dos Majoranas de una misma región (Jacobi), con el ángulo
         óptimo en forma cerrada, en precisión arbitraria. Hace falta por dos razones:
         Procrustes no puede mover un acople que quedó fuera de la diagonal entre dos
         posiciones con diagonal nula (le pesa cero), y la doble precisión deja ruido entre
         acoples de escalas muy distintas.
      4. Empaquetado: los acoples fuertes se juntan en las primeras posiciones (ver
         _empaquetar) y las posiciones se ordenan de mayor a menor acople. Solo cambia el
         orden, para que la matriz reducida muestre lo importante.

El resultado es un máximo local del peso diagonal: no hay garantía de que sea el global.
Las posiciones que no llegan a acoplarse con nada (por debajo de la precisión) se quedan en
su base de Williamson.

Archivos (en Resultados_diag_off/dps<n>, con <prefijo> como en diag_local):
    <prefijo>_off_Mtilde.txt      M_tilde = O M O^T, orden (a'_R1 | b'_R1 | a'_R2 | b'_R2 | ...)
    <prefijo>_off_O_<R>.txt       ortogonal local de la región R (orden (a_R | b_R))
    <prefijo>_off_info.txt        j antes y después, etapas, tiempo

Requiere: numpy, mpmath   (pip install numpy mpmath), y diag_local.py en la misma carpeta.
"""
import os
import time
import numpy as np
from mpmath import mp

DPS = 70
PROGRAMAS = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Procrustes alternado, en doble precisión (numpy)
# ---------------------------------------------------------------------------
def _polar(G):
    """La ortogonal más cercana a G: si G = U S V^T, es U V^T (Procrustes ortogonal)."""
    U, _, Vt = np.linalg.svd(G)
    return U @ Vt


def peso_diagonal(Wb, U, V):
    """(peso en la diagonal, peso total) de los bloques cruzados U_X^T W_XY V_Y, X != Y.
    Wb[X][Y]: bloques de W (numpy);  U[X], V[Y]: ortogonales con los nuevos Majoranas en las
    columnas."""
    F = total = 0.0
    for X in range(len(Wb)):
        for Y in range(len(Wb)):
            if X != Y and Wb[X][Y].size:
                t = U[X].T @ Wb[X][Y] @ V[Y]
                F += float((np.diag(t) ** 2).sum())
                total += float((t ** 2).sum())
    return F, total


def procrustes_alternado(Wb, U, V, max_iter=2000, tol=5e-15, mu=1e-12):
    """
    Procrustes alternado sobre los bloques cruzados. Cada paso actualiza una sola ortogonal:
        U_X <- polar( sum_{Y != X} W_XY V_Y D_XY^T )        (columna i: p_i d_i + ...)
        V_Y <- polar( sum_{X != Y} W_XY^T U_X D_XY )
    con D_XY = diagonal actual de U_X^T W_XY V_Y. El peso en la diagonal nunca baja.
    mu: a G se le suma mu * max|G| * (ortogonal actual), para que las columnas sin acople
    (columna nula de G) se queden como están.
    Criterio de parada: en el punto fijo, O^T G es simétrica. Se mide su parte antisimétrica
    (relativa a max|G|), que es proporcional a la rotación que todavía falta, y se para cuando
    baja de tol, o cuando deja de disminuir y el peso tampoco sube (ruido de la doble
    precisión). No basta con mirar el peso: deja de cambiar cuando la base apenas va por la
    mitad de las cifras.
    Devuelve (U, V, peso final, peso total, iteraciones).
    """
    n = len(Wb)
    U = [u.copy() for u in U]
    V = [v.copy() for v in V]

    def actualizar(O, G):
        escala = np.abs(G).max()
        if escala == 0:
            return O, 0.0                          # sin acople con nadie: se queda igual
        S = O.T @ G
        return _polar(G + mu * escala * O), np.abs(S - S.T).max() / escala

    residuos, pesos = [], []
    for it in range(1, max_iter + 1):
        residuo = peso = 0.0
        for X in range(n):
            if not U[X].size:
                continue
            G = np.zeros_like(U[X])
            for Y in range(n):
                if Y != X and Wb[X][Y].size:
                    P = Wb[X][Y] @ V[Y]                        # direcciones deseadas, en a_X
                    m = min(P.shape)
                    d = (U[X][:, :m] * P[:, :m]).sum(axis=0)   # acoples actuales
                    G[:, :m] += P[:, :m] * d
            U[X], r = actualizar(U[X], G)
            residuo = max(residuo, r)
        for Y in range(n):
            if not V[Y].size:
                continue
            G = np.zeros_like(V[Y])
            for X in range(n):
                if X != Y and Wb[X][Y].size:
                    Q = Wb[X][Y].T @ U[X]                      # direcciones deseadas, en b_Y
                    m = min(Q.shape)
                    d = (V[Y][:, :m] * Q[:, :m]).sum(axis=0)
                    G[:, :m] += Q[:, :m] * d
                    peso += float((d ** 2).sum())
            V[Y], r = actualizar(V[Y], G)
            residuo = max(residuo, r)
        residuos.append(residuo)
        pesos.append(peso)
        if residuo < tol:
            break
        if (it >= 40 and min(residuos[-20:]) > 0.5 * min(residuos[-40:-20])
                and pesos[-1] - pesos[-21] <= 1e-13 * pesos[-1]):
            break                                  # ni la base ni el peso mejoran ya
    F, total = peso_diagonal(Wb, U, V)
    return U, V, F, total, it


def _arranques(Wb):
    """Puntos de partida: la base actual y, para cada par de regiones, la SVD de sus dos
    bloques (W_XY y W_YX), con las demás regiones en la base actual."""
    n = len(Wb)
    identidad = lambda: ([np.eye(Wb[X][X].shape[0]) for X in range(n)],
                         [np.eye(Wb[X][X].shape[1]) for X in range(n)])
    lista = [("base actual", *identidad())]
    for X in range(n):
        for Y in range(X + 1, n):
            if not Wb[X][Y].size:
                continue
            U, V = identidad()
            u, _, vt = np.linalg.svd(Wb[X][Y])
            U[X], V[Y] = u, vt.T
            u, _, vt = np.linalg.svd(Wb[Y][X])
            U[Y], V[X] = u, vt.T
            lista.append((f"SVD del par {X}-{Y}", U, V))
    return lista


def mejor_procrustes(Wb, max_iter=2000, tol=5e-15):
    """Procrustes alternado desde cada arranque; devuelve el de mayor peso diagonal:
    (U, V, peso, total, nombre del arranque, iteraciones)."""
    mejor = None
    for nombre, U0, V0 in _arranques(Wb):
        U, V, F, total, it = procrustes_alternado(Wb, U0, V0, max_iter, tol)
        if mejor is None or F > mejor[2]:
            mejor = (U, V, F, total, nombre, it)
    return mejor


# ---------------------------------------------------------------------------
# Álgebra en precisión arbitraria, con listas de filas (más rápido que mp.matrix)
# ---------------------------------------------------------------------------
def _identidad(n):
    return [[mp.mpf(1 if i == j else 0) for j in range(n)] for i in range(n)]


def _producto(A, B):
    """A B (listas de filas)."""
    columnas = list(zip(*B))
    return [[mp.fdot(fila, col) for col in columnas] for fila in A]


def _ortonormalizar(Q):
    """Q (doble precisión, casi ortogonal) -> ortogonal exacta en precisión arbitraria, por
    Newton-Schulz:  X <- X (3 - X^T X) / 2  (el error se eleva al cuadrado en cada paso)."""
    n = len(Q)
    X = [[mp.mpf(float(x)) for x in fila] for fila in Q]
    if n == 0:
        return X
    pasos, digitos = 0, 14
    while digitos < mp.dps + 5:
        digitos *= 2
        pasos += 1
    for _ in range(pasos):
        Xt = [list(c) for c in zip(*X)]
        S = _producto(Xt, X)
        for i in range(n):
            for j in range(n):
                S[i][j] = (3 if i == j else 0) - S[i][j]
        X = [[x / 2 for x in fila] for fila in _producto(X, S)]
    return X


def _rotar_filas(T, filas, R):
    """Cambia las filas 'filas' de T por sus combinaciones con las columnas de R:
    nueva fila k = sum_l R[l][k] * fila l   (R ortogonal, nuevos vectores en las columnas)."""
    if not filas:
        return
    columnas = list(zip(*[T[f] for f in filas]))
    for k, f in enumerate(filas):
        coef = [R[l][k] for l in range(len(filas))]
        T[f] = [mp.fdot(coef, col) for col in columnas]


def _transponer(T):
    return [list(c) for c in zip(*T)]


def _pesos(T, tam, ini):
    """Peso diagonal de cada posición, sum_{X != Y} T_XY[i, i]^2, y los totales (F, total)."""
    n = len(tam)
    w = [mp.mpf(0)] * max(tam)
    total = mp.mpf(0)
    for X in range(n):
        for Y in range(n):
            if X == Y:
                continue
            for i in range(tam[X]):
                fila = T[ini[X] + i]
                total += mp.fsum(x * x for x in fila[ini[Y]:ini[Y] + tam[Y]])
                if i < tam[Y]:
                    w[i] += fila[ini[Y] + i] ** 2
    return w, mp.fsum(w), total


# ---------------------------------------------------------------------------
# Las tres partes del cálculo en precisión arbitraria
# ---------------------------------------------------------------------------
def _exacta_dos(T, A, B, tam, ini):
    """Dos regiones: SVD de los dos bloques cruzados. W_01 = U S V  ->  A_0 = U^T, B_1 = V."""
    for X, Y in ((0, 1), (1, 0)):
        bloque = mp.matrix([T[ini[X] + i][ini[Y]:ini[Y] + tam[Y]] for i in range(tam[X])])
        U, S, V = mp.svd_r(bloque, full_matrices=True)
        m = min(tam[X], tam[Y])
        orden = sorted(range(m), key=lambda i: -S[i])
        filas_a = orden + list(range(m, tam[X]))
        filas_b = orden + list(range(m, tam[Y]))
        A[X] = [[U[x, i] for x in range(tam[X])] for i in filas_a]
        B[Y] = [[V[i, x] for x in range(tam[Y])] for i in filas_b]


def _etapas(T, A, B, tam, ini, log, umbral=1e-4, max_etapas=80):
    """Procrustes alternado por etapas (ver el encabezado). Modifica T, A y B.
    umbral: en cada etapa quedan resueltas las posiciones con peso >= umbral * (peso máximo)."""
    n = len(tam)
    corte = mp.mpf(10) ** (-(mp.dps - 12))          # por debajo de esto no hay nada que resolver
    pendientes = list(range(max(tam)))              # posiciones sin resolver (comunes a todas)
    resumen = []
    for etapa in range(1, max_etapas + 1):
        pos = [[p for p in pendientes if p < tam[X]] for X in range(n)]
        if sum(1 for p in pos if p) < 2:
            break
        escala = mp.mpf(0)
        for X in range(n):
            for Y in range(n):
                if X != Y:
                    for i in pos[X]:
                        fila = T[ini[X] + i]
                        for j in pos[Y]:
                            escala = max(escala, abs(fila[ini[Y] + j]))
        if escala < corte:
            break
        # Los bloques entre posiciones pendientes, reescalados, en doble precisión
        Wb = [[np.array([[float(T[ini[X] + i][ini[Y] + j] / escala) for j in pos[Y]]
                         for i in pos[X]], dtype=float).reshape(len(pos[X]), len(pos[Y]))
               for Y in range(n)] for X in range(n)]
        U, V, F, total, arranque, it = mejor_procrustes(Wb)
        # Aplicar las rotaciones en precisión arbitraria
        for X in range(n):
            R = _ortonormalizar(U[X])
            _rotar_filas(T, [ini[X] + p for p in pos[X]], R)
            _rotar_filas(A[X], pos[X], R)
        Tt = _transponer(T)
        for Y in range(n):
            R = _ortonormalizar(V[Y])
            _rotar_filas(Tt, [ini[Y] + p for p in pos[Y]], R)
            _rotar_filas(B[Y], pos[Y], R)
        T[:] = _transponer(Tt)
        # Posiciones resueltas en esta etapa
        w = np.zeros(len(pendientes))
        for X in range(n):
            for Y in range(n):
                if X != Y and Wb[X][Y].size:
                    d = np.diag(U[X].T @ Wb[X][Y] @ V[Y])
                    w[:len(d)] += d ** 2
        resueltas = [p for k, p in enumerate(pendientes) if w[k] >= umbral * w.max()]
        pendientes = [p for p in pendientes if p not in resueltas]
        resumen.append(f"etapa {etapa}: escala {mp.nstr(escala, 3)}, {len(resueltas)} posiciones, "
                       f"fuera de la diagonal {max(0.0, 1 - F / total):.2e} ({arranque}, {it} it.)")
        log(resumen[-1])
        if not pendientes:
            break
    return resumen


def _barrido_jacobi(T, O, tam, ini, ruido, piso):
    """Un barrido de rotaciones de a dos filas de T dentro de cada región (las filas son los a'
    si T es W_tilde, o los b' si es su transpuesta). Para las filas i, j de la región X, el
    peso diagonal como función del ángulo es  alfa + beta cos 2t + delta sin 2t, con
        beta  = (1/2) sum_Y [ (x_ii^2 - x_ji^2) + (x_jj^2 - x_ij^2) ]
        delta = sum_Y [ x_ii x_ji - x_jj x_ij ]              (x = bloque XY, Y != X)
    y el máximo está en  tan 2t = delta / beta.
    No se rota si las cuatro entradas son puro ruido (ruido, relativo) ni si la rotación
    cambiaría las entradas en menos de 'piso'.
    Devuelve (número de rotaciones, mayor cambio |sin t| * |entradas| aplicado)."""
    n = len(tam)
    cero = mp.mpf(0)
    rotaciones, mayor = 0, cero
    for X in range(n):
        otras = [Y for Y in range(n) if Y != X]
        for i in range(tam[X] - 1):
            for j in range(i + 1, tam[X]):
                fi, fj = T[ini[X] + i], T[ini[X] + j]
                beta = delta = tamaño = cero
                for Y in otras:
                    c0 = ini[Y]
                    if i < tam[Y]:
                        xii, xji = fi[c0 + i], fj[c0 + i]
                        beta += xii * xii - xji * xji
                        delta += xii * xji
                        tamaño += xii * xii + xji * xji
                    if j < tam[Y]:
                        xjj, xij = fj[c0 + j], fi[c0 + j]
                        beta += xjj * xjj - xij * xij
                        delta -= xjj * xij
                        tamaño += xjj * xjj + xij * xij
                beta /= 2
                r = mp.sqrt(beta * beta + delta * delta)
                if r <= tamaño * ruido or tamaño == 0:
                    continue                              # nada que decidir entre estas dos
                # cos t y sin t con t en (-pi/2, pi/2], sin cancelaciones
                if beta >= 0:
                    mas = r + beta                        # r (1 + cos 2t)
                    menos = delta * delta / mas           # r (1 - cos 2t)
                else:
                    menos = r - beta
                    mas = delta * delta / menos
                c = mp.sqrt(mas / (2 * r))
                s = mp.sqrt(menos / (2 * r))
                if delta < 0:
                    s = -s
                efecto = abs(s) * mp.sqrt(tamaño)
                if efecto < piso:
                    continue
                T[ini[X] + i] = [c * a + s * b for a, b in zip(fi, fj)]
                T[ini[X] + j] = [c * b - s * a for a, b in zip(fi, fj)]
                oi, oj = O[X][i], O[X][j]
                O[X][i] = [c * a + s * b for a, b in zip(oi, oj)]
                O[X][j] = [c * b - s * a for a, b in zip(oi, oj)]
                rotaciones += 1
                mayor = max(mayor, efecto)
    return rotaciones, mayor


def _pulido(T, A, B, tam, ini, log, max_barridos=20, piso=1e-30, nombre="pulido"):
    """Pulido de Jacobi en precisión arbitraria: barridos sobre los a' y sobre los b' hasta que
    ninguna rotación cambie las entradas en más de 'piso' (o max_barridos). Modifica T, A y B."""
    ruido = mp.mpf(10) ** (-(mp.dps - 15))
    piso = max(mp.mpf(piso), mp.mpf(10) ** (-(mp.dps - 15)))
    hechos = 0
    for barrido in range(1, max_barridos + 1):
        r1, e1 = _barrido_jacobi(T, A, tam, ini, ruido, piso)
        Tt = _transponer(T)
        r2, e2 = _barrido_jacobi(Tt, B, tam, ini, ruido, piso)
        T[:] = _transponer(Tt)
        hechos = barrido
        log(f"{nombre}, barrido {barrido}: {r1 + r2} rotaciones, mayor cambio "
            f"{mp.nstr(max(e1, e2), 3)}")
        if r1 + r2 == 0:
            break
    return hechos


def _empaquetar(T, A, B, tam, ini, rel=1e-6):
    """
    Junta los acoples fuertes en las primeras posiciones, sin cambiar la solución.
    En cada posición hay hasta 2n Majoranas (a' y b' de cada región), unidos por las entradas
    diagonales de los bloques cruzados. Los que están unidos por entradas significativas
    forman un grupo que tiene que seguir compartiendo posición; grupos distintos se pueden
    mover por separado. Se recorren los grupos del más fuerte al más débil y cada uno va a la
    primera posición donde sus Majoranas caben. Así, por ejemplo, tres parejas que quedaron
    en tres posiciones distintas pasan a compartir la posición 0.
    Una entrada es significativa si su cuadrado es >= rel * (el mayor de los dos Majoranas).
    Si un grupo (siempre de los más débiles) no encuentra una posición libre para todos sus
    Majoranas, se suelta. Devuelve True si movió algo.
    """
    n = len(tam)
    ruido = mp.mpf(10) ** (-2 * (mp.dps - 12))
    grupos = []                                   # (fuerza, posición, ranuras)
    for p in range(max(tam)):                     # ranura 2X = a' de X, 2X + 1 = b' de X
        aristas = [(2 * X, 2 * Y + 1, T[ini[X] + p][ini[Y] + p] ** 2)
                   for X in range(n) for Y in range(n)
                   if X != Y and p < tam[X] and p < tam[Y]]
        mayor = {}
        for u, v, w in aristas:
            mayor[u] = max(mayor.get(u, 0), w)
            mayor[v] = max(mayor.get(v, 0), w)
        padre = list(range(2 * n))

        def raiz(u):
            while padre[u] != u:
                u = padre[u]
            return u

        fuerza = [mp.mpf(0)] * (2 * n)
        for u, v, w in aristas:
            if w > ruido and w >= rel * max(mayor[u], mayor[v]):
                ru, rv = raiz(u), raiz(v)
                if ru != rv:
                    padre[rv] = ru
                    fuerza[ru] += fuerza[rv]
                fuerza[ru] += w
        for r in range(2 * n):
            ranuras = [u for u in range(2 * n) if raiz(u) == r]
            if padre[r] == r and len(ranuras) > 1:
                grupos.append((fuerza[r], p, ranuras))
    grupos.sort(key=lambda g: -g[0])
    libre = [set(range(tam[r // 2])) for r in range(2 * n)]
    destino = [{} for _ in range(2 * n)]          # destino[ranura][posición nueva] = vieja
    for _, p, ranuras in grupos:
        comunes = set.intersection(*[libre[r] for r in ranuras])
        if not comunes:
            continue                              # no cabe junto: sus Majoranas van sueltos
        q = min(comunes)
        for r in ranuras:
            libre[r].remove(q)
            destino[r][q] = p
    orden = []
    for r in range(2 * n):
        usadas = set(destino[r].values())
        resto = [p for p in range(tam[r // 2]) if p not in usadas]
        for q, p in zip(sorted(libre[r]), resto):
            destino[r][q] = p
        orden.append([destino[r][q] for q in range(tam[r // 2])])
    if all(o == list(range(len(o))) for o in orden):
        return False
    filas = [ini[X] + p for X in range(n) for p in orden[2 * X]]
    columnas = [ini[Y] + p for Y in range(n) for p in orden[2 * Y + 1]]
    T[:] = [[T[f][c] for c in columnas] for f in filas]
    for X in range(n):
        A[X][:] = [A[X][p] for p in orden[2 * X]]
        B[X][:] = [B[X][p] for p in orden[2 * X + 1]]
    return True


def _ordenar(T, A, B, tam, ini):
    """Ordena las posiciones de mayor a menor peso diagonal (la misma permutación en todas las
    regiones). Si las regiones tienen tamaños distintos, solo se permuta entre posiciones que
    existen en las mismas regiones."""
    w, _, _ = _pesos(T, tam, ini)
    cortes = sorted(set([0] + list(tam)))
    orden = []
    for a, b in zip(cortes, cortes[1:]):
        orden += sorted(range(a, b), key=lambda p: -w[p])
    n = len(tam)
    filas = [ini[X] + p for X in range(n) for p in orden if p < tam[X]]
    T[:] = [T[f] for f in filas]
    T[:] = [[fila[c] for c in filas] for fila in T]
    for X in range(n):
        A[X][:] = [A[X][p] for p in orden if p < tam[X]]
        B[X][:] = [B[X][p] for p in orden if p < tam[X]]


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def diagonalizacion_off(M, regiones, O_inicial=None, pulido=True, max_barridos=20, log=print):
    """
    Diagonalización simultánea aproximada de los bloques cruzados entre las 'regiones'.

    M: matriz de covarianza (mp.matrix, 2N x 2N, orden (a_1..a_N | b_1..b_N)), quiral.
    regiones: lista de listas de sitios (0..N-1): las regiones que se diagonalizan entre sí,
        todas a la vez. Con dos regiones la solución es exacta.
    O_inicial: opcional, las ortogonales de Williamson de esas regiones (de diag_local), para
        no volver a calcularlas.
    pulido, max_barridos: pulido de Jacobi (solo con tres o más regiones). Sin él pueden
        quedar acoples fuera de la diagonal que Procrustes no sabe mover.

    Devuelve (M_tilde, O, info):
        M_tilde   O M O^T restringida a las regiones, orden (a'_R1 | b'_R1 | a'_R2 | ...)
        O         [O_R], con O_R = [[A_R, 0], [0, B_R]] y los nuevos Majoranas en las filas
        info      dict con j_inicial (base de Williamson), j_final, pesos, etapas, etc.
    """
    import diag_local
    t0 = time.time()
    N = M.rows // 2
    regiones = [list(R) for R in regiones]
    n = len(regiones)
    if n < 2:
        raise ValueError("Hacen falta al menos dos regiones.")
    tam = [len(R) for R in regiones]
    ini = [sum(tam[:X]) for X in range(n)]
    todos = [s for R in regiones for s in R]
    if len(set(todos)) != len(todos):
        raise ValueError("Las regiones no pueden compartir sitios.")
    quiral = max(max(abs(M[i, j]), abs(M[N + i, N + j])) for i in todos for j in todos)
    if quiral > mp.mpf(10) ** (-(mp.dps // 2)):
        raise ValueError(f"M no es quiral (bloques aa, bb hasta {mp.nstr(quiral, 3)}); "
                         "diag_off supone que solo hay correlaciones a-b.")

    # W en el orden de las regiones, y la base de partida: Williamson de cada región
    T = [[M[i, N + j] for j in todos] for i in todos]
    A, B = [], []
    for X, R in enumerate(regiones):
        O_R = O_inicial[X] if O_inicial is not None else diag_local.ortogonal_local(M, R, N)[0]
        A.append([[O_R[i, j] for j in range(tam[X])] for i in range(tam[X])])
        B.append([[O_R[tam[X] + i, tam[X] + j] for j in range(tam[X])] for i in range(tam[X])])
    aplicar_base(T, A, B, tam, ini)
    _, F, total = _pesos(T, tam, ini)
    info = dict(regiones=n, tamaños=tam, j_inicial=max(0.0, float(1 - F / total)),
                total=float(total))
    log(f"base de Williamson: fuera de la diagonal j = {info['j_inicial']:.3e}")

    if n == 2:
        T = [[M[i, N + j] for j in todos] for i in todos]
        _exacta_dos(T, A, B, tam, ini)
        aplicar_base(T, A, B, tam, ini)
        info["metodo"] = "exacto (SVD de los dos bloques)"
        info["etapas"] = []
    else:
        info["etapas"] = _etapas(T, A, B, tam, ini, log)
        info["metodo"] = "Procrustes alternado por etapas"
        _, F, _ = _pesos(T, tam, ini)
        info["j_procrustes"] = max(0.0, float(1 - F / total))
        log(f"después de Procrustes: j = {info['j_procrustes']:.3e}")
        if pulido:
            barridos = _pulido(T, A, B, tam, ini, log, max_barridos)
            if _empaquetar(T, A, B, tam, ini):     # mueve entradas muy pequeñas: se repasa
                barridos += _pulido(T, A, B, tam, ini, log, 5, nombre="repaso")
            info["barridos_pulido"] = barridos
            info["metodo"] += " + pulido de Jacobi"
        else:
            _empaquetar(T, A, B, tam, ini)
    _ordenar(T, A, B, tam, ini)
    w, F, _ = _pesos(T, tam, ini)
    info["j_final"] = max(0.0, float(1 - F / total))
    info["pesos"] = [float(x) for x in w]
    info["segundos"] = time.time() - t0
    log(f"resultado: j = {info['j_final']:.3e}   ({info['metodo']}, {info['segundos']:.0f} s)")

    # M_tilde y las ortogonales locales, en el formato de diag_local
    M_tilde = mp.zeros(2 * sum(tam), 2 * sum(tam))
    for X in range(n):
        for Y in range(n):
            for i in range(tam[X]):
                fila = T[ini[X] + i]
                for j in range(tam[Y]):
                    x = fila[ini[Y] + j]
                    M_tilde[2 * ini[X] + i, 2 * ini[Y] + tam[Y] + j] = x
                    M_tilde[2 * ini[Y] + tam[Y] + j, 2 * ini[X] + i] = -x
    O = []
    for X in range(n):
        O_R = mp.zeros(2 * tam[X], 2 * tam[X])
        for i in range(tam[X]):
            for j in range(tam[X]):
                O_R[i, j] = A[X][i][j]
                O_R[tam[X] + i, tam[X] + j] = B[X][i][j]
        O.append(O_R)
    return M_tilde, O, info


def aplicar_base(T, A, B, tam, ini):
    """T (W en la base de los sitios, por regiones) -> W_tilde_XY = A_X W_XY B_Y^T, en el sitio."""
    n = len(tam)
    for X in range(n):
        filas = list(range(ini[X], ini[X] + tam[X]))
        _rotar_filas(T, filas, _transponer(A[X]))
    Tt = _transponer(T)
    for Y in range(n):
        filas = list(range(ini[Y], ini[Y] + tam[Y]))
        _rotar_filas(Tt, filas, _transponer(B[Y]))
    T[:] = _transponer(Tt)


# ---------------------------------------------------------------------------
# Casos: leen M, calculan, guardan (o reutilizan si ya existe)
# ---------------------------------------------------------------------------
def _carpeta(dps):
    return os.path.join(PROGRAMAS, "Resultados_diag_off", f"dps{dps}")


def ya_calculado(prefijo, etiquetas, dps):
    archivos = [f"{prefijo}_off_Mtilde.txt", f"{prefijo}_off_info.txt"] + [
        f"{prefijo}_off_O_{r}.txt" for r in etiquetas]
    return all(os.path.exists(os.path.join(_carpeta(dps), a)) for a in archivos)


def leer_info(prefijo, dps):
    """El archivo de información de un caso guardado, como diccionario de textos."""
    info = {}
    with open(os.path.join(_carpeta(dps), f"{prefijo}_off_info.txt"), encoding="utf-8") as f:
        for linea in f:
            if ":" in linea:
                clave, valor = linea.split(":", 1)
                info[clave.strip()] = valor.strip()
    return info


def leer_Mtilde(prefijo, dps):
    """M_tilde de un caso guardado, en doble precisión (suficiente para graficar)."""
    archivo = os.path.join(_carpeta(dps), f"{prefijo}_off_Mtilde.txt")
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No encontré {prefijo}_off_Mtilde.txt en {_carpeta(dps)}")
    return np.loadtxt(archivo)


def guardar_resultado(prefijo, etiquetas, M_tilde, O, info, dps):
    import diag_local
    diag_local.DPS = dps
    os.makedirs(_carpeta(dps), exist_ok=True)
    diag_local.guardar_matriz(os.path.join(_carpeta(dps), f"{prefijo}_off_Mtilde.txt"), M_tilde)
    for r, O_R in zip(etiquetas, O):
        diag_local.guardar_matriz(os.path.join(_carpeta(dps), f"{prefijo}_off_O_{r}.txt"), O_R)
    with open(os.path.join(_carpeta(dps), f"{prefijo}_off_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"metodo: {info['metodo']}\n")
        f.write(f"j_inicial: {info['j_inicial']:.6e}\n")
        if "j_procrustes" in info:
            f.write(f"j_procrustes: {info['j_procrustes']:.6e}\n")
        f.write(f"j_final: {info['j_final']:.6e}\n")
        f.write(f"peso_total: {info['total']:.12g}\n")
        f.write(f"segundos: {info['segundos']:.0f}\n")
        for linea in info["etapas"]:
            f.write(linea.replace(":", " -", 1) + "\n")
        f.write("pesos_por_posicion: " + " ".join(f"{x:.6e}" for x in info["pesos"]) + "\n")


def diag_off_particion(gamma, h, N, regiones, etiquetas, sufijo, dps=None, M=None,
                       O_inicial=None, pulido=True, log=print):
    """Diagonalización off-diag de una partición en arcos (regiones, etiquetas y sufijo como los
    devuelven diag_local.triparticion y diag_local.biparticion). Si ya está guardada no la
    repite; si no, la calcula (leyendo M si no se la pasan) y la guarda.
    Devuelve (prefijo, info), con info como texto (ver leer_info)."""
    import diag_local
    global DPS
    if dps is not None:
        DPS = dps
    mp.dps = DPS
    prefijo = f"{diag_local.base_caso(gamma, h, N)}_{sufijo}"
    if ya_calculado(prefijo, etiquetas, DPS):
        log(f"{prefijo}: off-diag ya calculado en dps{DPS}.")
        return prefijo, leer_info(prefijo, DPS)
    if M is None:
        M = diag_local.leer_covarianza(gamma, h, N, DPS)
    M_tilde, O, info = diagonalizacion_off(M, regiones, O_inicial, pulido, log=log)
    guardar_resultado(prefijo, etiquetas, M_tilde, O, info, DPS)
    return prefijo, leer_info(prefijo, DPS)


def diag_off_caso(gamma, h, N, NA, NB, dps=None, log=print):
    """Tripartición A | B | C (C es el resto): los tres bloques cruzados a la vez."""
    import diag_local
    regiones, etiquetas, sufijo = diag_local.triparticion(N, NA, NB)
    return diag_off_particion(gamma, h, N, regiones, etiquetas, sufijo, dps, log=log)


def diag_off_biparticion_caso(gamma, h, N, NS, inicio=0, dps=None, log=print):
    """Bipartición S | T (S = arco de NS sitios desde 'inicio'): solución exacta."""
    import diag_local
    regiones, etiquetas, sufijo = diag_local.biparticion(N, inicio, NS)
    return diag_off_particion(gamma, h, N, regiones, etiquetas, sufijo, dps, log=log)


if __name__ == "__main__":
    gamma = 0.5
    h = 0.3
    N = 200
    NA, NB = 66, 66          # C es el resto: N - NA - NB
    dps = 70

    diag_off_caso(gamma, h, N, NA, NB, dps)
