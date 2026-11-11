"""
Consola (ventana) para explorar la cadena XY en anillo.

Cada botón encadena los programas que hagan falta:
    1. hamiltoniano            -> Resultados_Ham_Solver/dps<n>
    2. matriz de covarianza    -> Resultados_covariance_matrix/dps<n>
    3. diagonalización local   -> Resultados_diag_local/dps<n>
    4. gráfica                 -> Resultados_modos  o  Resultados_matrices

Pestaña "Tripartito" (gamma, h, N, NA, NB, NC, dps):
    Graficar anillo          gráfica cilíndrica de A | B | C   (graf_modos)
    Graficar matriz          W_tilde reducida, escala log y lineal   (graf_matrix)
    Graficar grafos          grafos de acoplamientos, modos y Majoranas   (graf_matrix)
    Biparticiones asociadas  A|BC, B|CA y AB|C: gráfica cilíndrica y W_tilde reducida de cada una
Pestaña "Bipartito" (gamma, h, N, NS, dps), con S = sitios 1..NS y T = el resto:
    Graficar modos           gráfica cilíndrica de S | T
    Graficar matriz          W_tilde reducida, escala log y lineal
Pestaña "Barrido": recorre varios valores de h, de gamma o de la partición (uno a la vez)
y arma una sola gráfica cilíndrica con un deslizador sobre ese parámetro.
Pestaña "Diagrama de fases" (N y partición fijos): diagrama (gamma, h) con un punto por caso.
    Al pasar el mouse muestra las fotos del caso más cercano (cilindro y W_tilde reducida);
    al hacer clic abre el cilindro 3D. Reutiliza lo ya calculado (ver elegir_dps) y calcula
    con dps_nuevo los puntos de la malla que falten. Las fotos van a Resultados_fotos.

Si un resultado ya existe en su carpeta dps<n>, se reutiliza en lugar de recalcularlo.

Uso: poner este archivo en la carpeta Programas, junto a los demás, y ejecutarlo.
Requiere: mpmath, numpy, plotly, scipy (y tkinter, que viene con Python en Windows).
Para las fotos del diagrama de fases: kaleido (pip install kaleido), que usa Google Chrome.
"""
import importlib
import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from decimal import Decimal, InvalidOperation
from pathlib import Path

import tkinter as tk
from tkinter import ttk

# Nombres de los programas (sin .py). Cámbialos aquí si tus archivos se llaman distinto.
MODULOS = {
    "ham": "ham_solver",
    "cov": "covariance_matrix",
    "diag": "diag_local",
    "graf": "graf_modos",
    "mat": "graf_matrix",
}

PROGRAMAS = os.path.dirname(os.path.abspath(__file__))

# Tiempos aproximados de cada paso con N = 200 y dps = 70 (segundos), medidos; solo para estimar.
# Escalan como (N/200)^3.
T_HAM, T_COV, T_DIAG, T_GRAF = 300.0, 45.0, 60.0, 5.0


# ---------------------------------------------------------------------------
# Cálculo (independiente de la ventana)
# ---------------------------------------------------------------------------
def _modulo(clave):
    if PROGRAMAS not in sys.path:
        sys.path.insert(0, PROGRAMAS)
    return importlib.import_module(MODULOS[clave])


def _carpeta(nombre, dps, crear=True):
    ruta = os.path.join(PROGRAMAS, nombre, f"dps{dps}")
    if crear:
        os.makedirs(ruta, exist_ok=True)
    return ruta


def _otras_precisiones(nombre_carpeta, archivo):
    """dps<n> de otras carpetas donde ya existe ese archivo (solo para informar)."""
    base = os.path.join(PROGRAMAS, nombre_carpeta)
    if not os.path.isdir(base):
        return []
    return [d for d in sorted(os.listdir(base))
            if os.path.isfile(os.path.join(base, d, archivo))]


def _nombres(gamma, h, N, NA, NB, NC):
    base = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}"
    return base, f"{base}_A{NA}_B{NB}_C{NC}"


def estado_caso(gamma, h, N, NA, NB, NC, dps):
    """Qué pasos de un caso ya existen: (ham, cov, diag) como booleanos."""
    base, prefijo = _nombres(gamma, h, N, NA, NB, NC)
    ham = os.path.exists(os.path.join(_carpeta("Resultados_Ham_Solver", dps, False),
                                      base + "_O_tilde.txt"))
    cov = os.path.exists(os.path.join(_carpeta("Resultados_covariance_matrix", dps, False),
                                      base + "_M.txt"))
    carpeta = _carpeta("Resultados_diag_local", dps, False)
    diag = all(os.path.exists(os.path.join(carpeta, f"{prefijo}_O_{r}.txt")) for r in "ABC")
    return ham, cov, diag


def tiempo_estimado(N, ham, cov, diag):
    """Segundos aproximados que faltan para un caso, según los pasos que ya existen."""
    escala = (N / 200.0) ** 3
    t = T_GRAF
    if not diag:
        t += T_DIAG * escala
        if not cov:
            t += T_COV * escala
            if not ham:
                t += T_HAM * escala
    return t


def preparar_covarianza(gamma, h, N, dps, log=print):
    """Pasos 1 y 2: devuelve la matriz de covarianza M (calcula lo que falte)."""
    ham, cov = _modulo("ham"), _modulo("cov")
    from mpmath import mp
    mp.dps = dps
    cov.DPS = dps
    base = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}"

    # 1. Hamiltoniano
    carpeta = _carpeta("Resultados_Ham_Solver", dps)
    archivo = os.path.join(carpeta, base + "_O_tilde.txt")
    archivo_M = os.path.join(_carpeta("Resultados_covariance_matrix", dps), base + "_M.txt")
    Ot = None
    if os.path.exists(archivo_M):
        log(f"   [1/4] Hamiltoniano: no hace falta (la covarianza ya existe).")
    elif os.path.exists(archivo):
        log(f"   [1/4] Hamiltoniano: ya existe en dps{dps}, lo leo.")
        Ot = cov.leer_matriz(archivo)
    else:
        otras = _otras_precisiones("Resultados_Ham_Solver", base + "_O_tilde.txt")
        nota = f" (existe con {', '.join(otras)}, pero no con dps{dps})" if otras else ""
        log(f"   [1/4] Hamiltoniano: calculando con dps{dps}{nota}...")
        t0 = time.time()
        Ot, Lam, E0 = ham.ortogonal_anillo_par(float(gamma), float(h), dps, N)
        ham.guardar(os.path.join(carpeta, base), Ot, Lam, dps)
        log(f"         listo en {time.time() - t0:.0f} s,  E0/N = {mp.nstr(E0 / N, 12)}")
    mp.dps = dps

    # 2. Matriz de covarianza
    if os.path.exists(archivo_M):
        log(f"   [2/4] Matriz de covarianza: ya existe en dps{dps}, la leo.")
        return cov.leer_matriz(archivo_M)
    log(f"   [2/4] Matriz de covarianza: calculando...")
    t0 = time.time()
    M = cov.covarianza(Ot)
    cov.guardar_matriz(archivo_M, M)
    log(f"         lista en {time.time() - t0:.0f} s")
    return M


def preparar_particiones(gamma, h, N, particiones, dps, log=print):
    """Pasos 1 a 3 para varias particiones del mismo caso (la covarianza se lee una sola vez).
    particiones: lista de (nombre, regiones, etiquetas, sufijo), con (regiones, etiquetas, sufijo)
    como los devuelven diag_local.triparticion y diag_local.biparticion.
    Devuelve, para cada partición, la lista de sus ortogonales locales."""
    diag = _modulo("diag")
    from mpmath import mp
    mp.dps = dps
    diag.DPS = dps
    base = diag.base_caso(gamma, h, N)
    carpeta = _carpeta("Resultados_diag_local", dps)
    resultado, M = [], None
    for nombre, regiones, etiquetas, sufijo in particiones:
        prefijo = f"{base}_{sufijo}"
        if diag.ya_calculado(prefijo, etiquetas, dps):
            log(f"   [1-3] {nombre}: ya calculado en dps{dps}, lo leo.")
            resultado.append([diag.leer_matriz(os.path.join(carpeta, f"{prefijo}_O_{r}.txt"))
                              for r in etiquetas])
            continue
        if M is None:
            M = preparar_covarianza(gamma, h, N, dps, log)
        log(f"   [3/4] Diagonalización local {nombre}: calculando...")
        _, O, _, _ = diag.diag_particion(gamma, h, N, regiones, etiquetas, sufijo, dps, M=M,
                                         log=lambda m: log("         " + m))
        resultado.append(O)
    return resultado


def preparar_caso(gamma, h, N, NA, NB, NC, dps, log=print):
    """Pasos 1 a 3 de la tripartición: devuelve las ortogonales locales [O_A, O_B, O_C]."""
    if NA + NB + NC != N:
        raise ValueError(f"NA + NB + NC = {NA + NB + NC}, pero N = {N}.")
    regiones, etiquetas, sufijo = _modulo("diag").triparticion(N, NA, NB)
    return preparar_particiones(gamma, h, N, [(f"A|B|C = {NA}|{NB}|{NC}", regiones, etiquetas,
                                               sufijo)], dps, log)[0]


def preparar_biparticion(gamma, h, N, NS, inicio, dps, log=print, nombre="S|T"):
    """Pasos 1 a 3 de la bipartición S | T: devuelve [O_S, O_T]."""
    regiones, etiquetas, sufijo = _modulo("diag").biparticion(N, inicio, NS)
    return preparar_particiones(gamma, h, N, [(f"{nombre} = {NS}|{N - NS}", regiones, etiquetas,
                                               sufijo)], dps, log)[0]


def regiones_para_grafica(O, N, NA, NB, NC, max_modos=None):
    """(nombre, H, C, sitios) de cada región, recortadas a max_modos para ahorrar memoria."""
    graf = _modulo("graf")
    sitios = [range(1, NA + 1), range(NA + 1, NA + NB + 1), range(NA + NB + 1, N + 1)]
    regiones = []
    for r, O_R, s in zip("ABC", O, sitios):
        H, C = graf.funcion_de_modos(O_R)
        if max_modos is not None:
            H, C = H[:max_modos], C[:max_modos]
        regiones.append((r, H, C, list(s)))
    return regiones


def _cilindro(regiones, N, archivo, titulo, max_modos, refinar, log):
    _modulo("graf").graficar_cilindro(regiones, N, max_modos=max_modos, refinar=refinar,
                                      archivo=archivo, titulo=titulo)
    log(f"         guardada en {os.path.basename(os.path.dirname(archivo))}/"
        f"{os.path.basename(archivo)}")
    return archivo


def _anotar(archivos, log):
    for a in archivos:
        log(f"         guardada en {os.path.basename(os.path.dirname(a))}/{os.path.basename(a)}")
    return archivos


# Cada botón llama a una de estas funciones; todas devuelven la lista de .html guardados.
def correr_caso(gamma, h, N, NA, NB, NC, dps, max_modos=None, refinar=1, log=print, **_):
    """Tripartito, "Graficar anillo": gráfica cilíndrica de A | B | C."""
    O = preparar_caso(gamma, h, N, NA, NB, NC, dps, log)
    log("   [4/4] Gráfica cilíndrica...")
    _, prefijo = _nombres(gamma, h, N, NA, NB, NC)
    salida = os.path.join(PROGRAMAS, "Resultados_modos")
    os.makedirs(salida, exist_ok=True)
    html = os.path.join(salida, f"{prefijo}_dps{dps}_ABC_cilindro.html")
    return [_cilindro(regiones_para_grafica(O, N, NA, NB, NC), N, html,
                      f"Anillo, regiones A | B | C   (γ={gamma}, h={h}, N={N}, "
                      f"A|B|C = {NA}|{NB}|{NC}, dps={dps})", max_modos, refinar, log)]


def correr_matriz(gamma, h, N, NA, NB, NC, dps, k=8, log=print, **_):
    """Tripartito, "Graficar matriz": W_tilde reducida (log y lineal)."""
    preparar_caso(gamma, h, N, NA, NB, NC, dps, log)
    log(f"   [4/4] Matriz reducida ({k} modos por región)...")
    return _anotar(_modulo("mat").graficar_caso(gamma, h, N, NA, NB, dps, k), log)


def correr_grafos(gamma, h, N, NA, NB, NC, dps, log=print, **_):
    """Tripartito, "Graficar grafos": grafos de acoplamientos (modos y Majoranas)."""
    preparar_caso(gamma, h, N, NA, NB, NC, dps, log)
    log("   [4/4] Grafos de acoplamientos...")
    return _anotar(_modulo("mat").graficar_caso_grafos(gamma, h, N, NA, NB, dps), log)


def _graficas_biparticion(O, gamma, h, N, NS, inicio, dps, etiquetas, que, k, max_modos,
                          refinar, log):
    """Gráficas de una bipartición ya calculada. que: "modos", "matriz" o ambos."""
    archivos = []
    nombre = f"{etiquetas[0]} | {etiquetas[1]}"
    if "modos" in que:
        log(f"   [4/4] {nombre}: gráfica cilíndrica...")
        diag = _modulo("diag")
        prefijo = f"{diag.base_caso(gamma, h, N)}_{diag.biparticion(N, inicio, NS)[2]}"
        salida = os.path.join(PROGRAMAS, "Resultados_modos")
        os.makedirs(salida, exist_ok=True)
        inicio %= N
        sitios = [list(range(inicio + 1, inicio + NS + 1)),        # sin dar la vuelta: si el
                  list(range(inicio + NS + 1, inicio + N + 1))]    # arco cruza (N, 1), sigue en N+1
        graf = _modulo("graf")
        regiones = []
        for r, O_R, s in zip(etiquetas, O, sitios):
            H, C = graf.funcion_de_modos(O_R)
            if max_modos is not None:
                H, C = H[:max_modos], C[:max_modos]
            regiones.append((r, H, C, s))
        archivos.append(_cilindro(
            regiones, N, os.path.join(salida, f"{prefijo}_dps{dps}_ST_cilindro.html"),
            f"Anillo, bipartición {nombre}   (γ={gamma}, h={h}, N={N}, {nombre} = {NS}|{N - NS}, "
            f"{etiquetas[0]} desde el sitio {inicio + 1}, dps={dps})", max_modos, refinar, log))
    if "matriz" in que:
        log(f"   [4/4] {nombre}: matriz reducida ({k} modos por región)...")
        archivos += _anotar(_modulo("mat").graficar_caso_biparticion(gamma, h, N, NS, inicio, dps,
                                                                     k, tuple(etiquetas)), log)
    return archivos


def correr_asociadas(gamma, h, N, NA, NB, NC, dps, k=8, max_modos=None, refinar=1, log=print,
                     **_):
    """Tripartito, "Biparticiones asociadas": A|BC, B|CA y AB|C, cada una con su gráfica
    cilíndrica y su W_tilde reducida."""
    if NA + NB + NC != N:
        raise ValueError(f"NA + NB + NC = {NA + NB + NC}, pero N = {N}.")
    diag = _modulo("diag")
    asociadas = diag.biparticiones_asociadas(N, NA, NB)
    particiones = [(nombre, *diag.biparticion(N, inicio, NS))
                   for nombre, inicio, NS in asociadas]
    Os = preparar_particiones(gamma, h, N, particiones, dps, log)
    archivos = []
    for (nombre, inicio, NS), O in zip(asociadas, Os):
        archivos += _graficas_biparticion(O, gamma, h, N, NS, inicio, dps, nombre.split("|"),
                                          ("modos", "matriz"), k, max_modos, refinar, log)
    return archivos


def correr_bip_modos(gamma, h, N, NS, dps, max_modos=None, refinar=1, log=print, **_):
    """Bipartito, "Graficar modos": gráfica cilíndrica de S | T, con S = sitios 1..NS."""
    O = preparar_biparticion(gamma, h, N, NS, 0, dps, log)
    return _graficas_biparticion(O, gamma, h, N, NS, 0, dps, ("S", "T"), ("modos",), None,
                                 max_modos, refinar, log)


def correr_bip_matriz(gamma, h, N, NS, dps, k=8, log=print, **_):
    """Bipartito, "Graficar matriz": W_tilde reducida (log y lineal) de S | T."""
    O = preparar_biparticion(gamma, h, N, NS, 0, dps, log)
    return _graficas_biparticion(O, gamma, h, N, NS, 0, dps, ("S", "T"), ("matriz",), k,
                                 None, 1, log)


# ---------------------------------------------------------------------------
# Barrido
# ---------------------------------------------------------------------------
def leer_numeros(texto):
    """'0.1, 0.3' -> [0.1, 0.3];  '0:1.5:0.25' -> [0.0, 0.25, ..., 1.5] (incluye el final).
    Usa aritmética decimal para que 0.1 + 0.2 dé 0.3 y no 0.30000000000000004."""
    texto = texto.strip()
    try:
        if ":" in texto:
            inicio, fin, paso = (Decimal(x.strip()) for x in texto.split(":"))
            if paso <= 0:
                raise ValueError("El paso tiene que ser positivo.")
            valores, v = [], inicio
            while v <= fin:
                valores.append(float(v))
                v += paso
            return valores
        return [float(Decimal(x.strip())) for x in texto.replace(";", ",").split(",") if x.strip()]
    except InvalidOperation:
        raise ValueError(f"No entiendo los valores '{texto}'.")


def leer_particion(texto):
    """'66|66|68' -> (66, 66, 68)."""
    partes = [p.strip() for p in texto.strip().split("|")]
    if len(partes) != 3:
        raise ValueError(f"La partición '{texto}' debe tener la forma NA|NB|NC.")
    return tuple(int(p) for p in partes)


def leer_particiones(texto):
    """'66|66|68; 50|100|50' -> [(66, 66, 68), (50, 100, 50)]."""
    return [leer_particion(t) for t in texto.split(";") if t.strip()]


def dps_de(h, dps, dps_critico):
    """Precisión de un caso: dps_critico en h = ±1 (si se pidió), dps en el resto."""
    return dps_critico if (dps_critico and abs(float(h)) == 1.0) else dps


def armar_barrido(parametro, valores, gamma, h, N, particion, dps, dps_critico):
    """Lista de casos (dict) del barrido. parametro: 'h', 'γ' o 'partición'."""
    casos = []
    if parametro == "h":
        lista = [(gamma, v, particion, f"h={v}") for v in leer_numeros(valores)]
    elif parametro == "γ":
        lista = [(v, h, particion, f"γ={v}") for v in leer_numeros(valores)]
    else:
        lista = [(gamma, h, p, f"{p[0]}|{p[1]}|{p[2]}") for p in leer_particiones(valores)]
    for g, hh, (NA, NB, NC), etiqueta in lista:
        if NA + NB + NC != N:
            raise ValueError(f"La partición {NA}|{NB}|{NC} no suma N = {N}.")
        casos.append(dict(gamma=g, h=hh, N=N, NA=NA, NB=NB, NC=NC,
                          dps=dps_de(hh, dps, dps_critico), etiqueta=etiqueta))
    if not casos:
        raise ValueError("El barrido no tiene ningún valor.")
    return casos


TODO = "todo lo calculado"


def buscar_calculados(N):
    """Todos los hamiltonianos guardados para N, en Resultados_Ham_Solver/dps*/.
    Si un mismo (gamma, h) está con varias precisiones, se usa la mayor.
    Devuelve una lista ordenada de (gamma, h, dps)."""
    import re
    base = os.path.join(PROGRAMAS, "Resultados_Ham_Solver")
    patron = re.compile(rf"anillo_par_N{N}_gamma([-\d.eE]+)_h([-\d.eE]+)_O_tilde\.txt$")
    encontrados = {}
    if os.path.isdir(base):
        for carpeta in os.listdir(base):
            m_dps = re.fullmatch(r"dps(\d+)", carpeta)
            if not m_dps:
                continue
            dps = int(m_dps.group(1))
            for nombre in os.listdir(os.path.join(base, carpeta)):
                m = patron.match(nombre)
                if m:
                    clave = (float(m.group(1)), float(m.group(2)))
                    encontrados[clave] = max(dps, encontrados.get(clave, 0))
    return sorted((g, h, d) for (g, h), d in encontrados.items())


def casos_calculados(N, particion):
    """Casos de 'todo lo calculado' con una misma partición, cada uno con su precisión."""
    NA, NB, NC = particion
    if NA + NB + NC != N:
        raise ValueError(f"La partición {NA}|{NB}|{NC} no suma N = {N}.")
    casos = [dict(gamma=g, h=h, N=N, NA=NA, NB=NB, NC=NC, dps=d, etiqueta=f"h={h}")
             for g, h, d in buscar_calculados(N)]
    if not casos:
        raise ValueError(f"No encontré hamiltonianos guardados con N = {N} en "
                         "Resultados_Ham_Solver/dps*/.")
    return casos


def nombre_barrido(parametro, casos):
    c0, c1 = casos[0], casos[-1]
    if parametro == "h":
        fijo, rango = f"gamma{c0['gamma']}_A{c0['NA']}_B{c0['NB']}_C{c0['NC']}", f"h{c0['h']}-{c1['h']}"
    elif parametro == "γ":
        fijo, rango = f"h{c0['h']}_A{c0['NA']}_B{c0['NB']}_C{c0['NC']}", f"gamma{c0['gamma']}-{c1['gamma']}"
    else:
        fijo, rango = f"gamma{c0['gamma']}_h{c0['h']}", f"particiones{len(casos)}"
    dpss = sorted({c["dps"] for c in casos})
    dps = f"dps{dpss[0]}" if len(dpss) == 1 else f"dps{dpss[0]}-{dpss[-1]}"
    return f"barrido_N{c0['N']}_{fijo}_{rango}_{dps}.html"


# ---------------------------------------------------------------------------
# Diagrama de fases
# ---------------------------------------------------------------------------
DPS_EXACTO = 10      # dps < 10: caso con números exactos (p. ej. gamma = 1, h = 0); se acepta
TAM_FOTO = (400, 380)  # ancho y alto (píxeles) de cada panel del diagrama de fases


def es_critico(gamma, h):
    """Líneas críticas: h = 1 (todo gamma) y gamma = 0 con h <= 1 (XX)."""
    return abs(h) == 1.0 or (gamma == 0.0 and abs(h) <= 1.0)


def buscar_precisiones(N, NA, NB, NC):
    """{(gamma, h): {dps, ...}} con todo lo guardado de este N y esta partición: hamiltoniano,
    covarianza o diagonalización local (lo que falte se completa con esa misma precisión)."""
    import re
    num = r"([-\d.eE]+)"
    patrones = {
        "Resultados_Ham_Solver": rf"anillo_par_N{N}_gamma{num}_h{num}_O_tilde\.txt$",
        "Resultados_covariance_matrix": rf"anillo_par_N{N}_gamma{num}_h{num}_M\.txt$",
        "Resultados_diag_local": rf"anillo_par_N{N}_gamma{num}_h{num}_A{NA}_B{NB}_C{NC}_Mtilde\.txt$",
    }
    encontrados = {}
    for carpeta, patron in patrones.items():
        base = os.path.join(PROGRAMAS, carpeta)
        if not os.path.isdir(base):
            continue
        patron = re.compile(patron)
        for sub in os.listdir(base):
            m_dps = re.fullmatch(r"dps(\d+)", sub)
            if not m_dps or not os.path.isdir(os.path.join(base, sub)):
                continue
            for nombre in os.listdir(os.path.join(base, sub)):
                m = patron.match(nombre)
                if m:
                    clave = (float(m.group(1)), float(m.group(2)))
                    encontrados.setdefault(clave, set()).add(int(m_dps.group(1)))
    return encontrados


def elegir_dps(precisiones, gamma, h, dps_min=70, dps_min_critico=100):
    """La mayor precisión aceptable entre las guardadas: >= dps_min (>= dps_min_critico en las
    líneas críticas), o un caso exacto (dps < DPS_EXACTO). None si no hay ninguna."""
    minimo = dps_min_critico if es_critico(gamma, h) else dps_min
    validos = [d for d in precisiones if d < DPS_EXACTO or d >= minimo]
    return max(validos) if validos else None


def puntos_diagrama(N, NA, NB, NC, gammas, hs, dps_nuevo=150, dps_min=70, dps_min_critico=100):
    """Puntos del diagrama: los ya guardados (con precisión aceptable) y los de la malla
    gammas x hs. Cada punto: dict(gamma, h, dps, listo), con listo = diagonalización local
    hecha. Los puntos de la malla sin nada aceptable guardado llevan dps = dps_nuevo."""
    guardados = buscar_precisiones(N, NA, NB, NC)
    malla = {(float(g), float(h)) for g in gammas for h in hs}
    puntos = []
    for g, h in sorted(set(guardados) | malla):
        d = elegir_dps(guardados.get((g, h), ()), g, h, dps_min, dps_min_critico)
        if d is None:
            if (g, h) not in malla:
                continue
            d = dps_nuevo
        listo = estado_caso(g, h, N, NA, NB, NC, d)[2] and os.path.exists(os.path.join(
            _carpeta("Resultados_diag_local", d, False), f"{_nombres(g, h, N, NA, NB, NC)[1]}_Mtilde.txt"))
        puntos.append(dict(gamma=g, h=h, dps=d, listo=listo))
    return puntos


def rutas_caso(gamma, h, N, NA, NB, NC, dps, max_modos, k):
    """(foto del cilindro, foto de la matriz reducida, cilindro 3D .html) de un punto."""
    _, prefijo = _nombres(gamma, h, N, NA, NB, NC)
    fotos = os.path.join(PROGRAMAS, "Resultados_fotos")
    return (os.path.join(fotos, f"{prefijo}_dps{dps}_cilindro_m{max_modos}.png"),
            os.path.join(fotos, f"{prefijo}_dps{dps}_W_reducida_k{k}.png"),
            os.path.join(PROGRAMAS, "Resultados_modos", f"{prefijo}_dps{dps}_ABC_cilindro.html"))


def preparar_fotos(gamma, h, N, NA, NB, NC, dps, max_modos=30, k=8, log=print):
    """Fotos (PNG, del tamaño TAM_FOTO) del cilindro y de la W_tilde reducida (escala log) de un
    caso ya diagonalizado. Solo hace las que falten. Requiere kaleido."""
    foto_cil, foto_mat, _ = rutas_caso(gamma, h, N, NA, NB, NC, dps, max_modos, k)
    os.makedirs(os.path.dirname(foto_cil), exist_ok=True)
    ancho, alto = TAM_FOTO
    if not os.path.exists(foto_cil):
        O = preparar_caso(gamma, h, N, NA, NB, NC, dps, log=lambda m: None)
        fig = _modulo("graf").figura_cilindro(regiones_para_grafica(O, N, NA, NB, NC, max_modos),
                                              N, max_modos=max_modos)
        fig.update_layout(title=None, width=ancho, height=alto, margin=dict(l=0, r=0, t=0, b=0),
                          font=dict(size=10))
        fig.write_image(foto_cil)
    if not os.path.exists(foto_mat):
        mat = _modulo("mat")
        M_tilde, _, _ = mat.leer_caso(gamma, h, N, NA, NB, dps)
        Wk, kk = mat.reducir(mat.W_tilde(M_tilde, NA, NB, NC), NA, NB, NC, k)
        fig = mat.figura_reducida(Wk, kk)
        fig.update_layout(title=None, width=ancho, height=alto, font=dict(size=9),
                          margin=dict(l=40, r=10, t=40, b=10),
                          xaxis=dict(title=None), yaxis=dict(title=None))
        fig.update_traces(colorbar=dict(thickness=10, title=None))
        fig.write_image(foto_mat)
    return foto_cil, foto_mat


def correr_cilindro_punto(gamma, h, N, NA, NB, NC, dps, max_modos=30, log=print, **_):
    """Clic en el diagrama: el cilindro 3D del punto. Reutiliza el .html si ya existe."""
    html = rutas_caso(gamma, h, N, NA, NB, NC, dps, max_modos, 0)[2]
    if os.path.exists(html):
        log(f"   Ya existe: Resultados_modos/{os.path.basename(html)}")
        return [html]
    return correr_caso(gamma, h, N, NA, NB, NC, dps, max_modos=max_modos, log=log)


# ---------------------------------------------------------------------------
# Ventana
# ---------------------------------------------------------------------------
def _formatear_tiempo(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


class Consola(tk.Tk):
    CAMPOS_CASO = [  # (clave, etiqueta, valor inicial)
        ("gamma", "γ", "0.5"),
        ("h", "h", "0.3"),
        ("N", "N", "200"),
        ("NA", "N_A", "66"),
        ("NB", "N_B", "66"),
        ("NC", "N_C", "68"),
        ("dps", "dps", "70"),
        ("max_modos", "máx. modos (vacío = todos)", ""),
        ("refinar", "refinar", "1"),
        ("k", "k (modos por región, matriz)", "8"),
    ]
    CAMPOS_BIP = [
        ("gamma", "γ", "0.5"),
        ("h", "h", "0.3"),
        ("N", "N", "200"),
        ("NS", "N_S", "100"),
        ("dps", "dps", "70"),
        ("max_modos", "máx. modos (vacío = todos)", ""),
        ("refinar", "refinar", "1"),
        ("k", "k (modos por región, matriz)", "8"),
    ]

    def __init__(self):
        super().__init__()
        self.title("Consola XY: anillo, tripartición y bipartición")
        self.geometry("1320x900")
        self.minsize(680, 600)
        self.cola = queue.Queue()
        self.ultimo_html = None
        self.ocupado = False
        self.detener = threading.Event()

        cuerpo = ttk.Frame(self, padding=12)
        cuerpo.pack(fill="both", expand=True)
        pestañas = ttk.Notebook(cuerpo)
        pestañas.pack(fill="x")
        self.botones_calculo = []
        self._pestaña_caso(pestañas)
        self._pestaña_bip(pestañas)
        self._pestaña_barrido(pestañas)
        self._pestaña_fases(pestañas)

        abajo = ttk.Frame(cuerpo)
        abajo.pack(fill="x", pady=(10, 6))
        self.boton_abrir = ttk.Button(abajo, text="Abrir última gráfica",
                                      command=self._abrir_ultima, state="disabled")
        self.boton_abrir.pack(side="left")
        ttk.Button(abajo, text="Limpiar registro", command=self._limpiar).pack(side="left", padx=8)

        registro = ttk.LabelFrame(cuerpo, text="Registro", padding=6)
        registro.pack(fill="both", expand=True)
        self.texto = tk.Text(registro, height=10, wrap="word", font=("Consolas", 10),
                             state="disabled")
        barra = ttk.Scrollbar(registro, command=self.texto.yview)
        self.texto.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)

        self._revisar_suma()
        self._actualizar_fijos()
        self._buscar_fases(silencioso=True)
        self.after(100, self._leer_cola)

    # --- pestañas "Tripartito" y "Bipartito" -------------------------------
    def _campos(self, marco, campos, al_cambiar):
        param = ttk.LabelFrame(marco, text="Parámetros", padding=10)
        param.pack(fill="x")
        variables = {}
        for k, (clave, etiqueta, valor) in enumerate(campos):
            fila, col = divmod(k, 3)
            ttk.Label(param, text=etiqueta).grid(row=fila, column=2 * col, sticky="e",
                                                 padx=(10 if col else 0, 4), pady=4)
            v = tk.StringVar(value=valor)
            ttk.Entry(param, textvariable=v, width=10).grid(row=fila, column=2 * col + 1,
                                                             sticky="w", pady=4)
            variables[clave] = v
            v.trace_add("write", lambda *_: al_cambiar())
        aviso = ttk.Label(param, text="")
        aviso.grid(row=(len(campos) + 2) // 3, column=0, columnspan=6, sticky="w", pady=(6, 0))
        return variables, aviso

    def _botones(self, marco, botones):
        fila = ttk.Frame(marco)
        fila.pack(fill="x", pady=(10, 0))
        for texto, comando in botones:
            b = ttk.Button(fila, text=texto, command=comando)
            b.pack(side="left", padx=(0, 8))
            self.botones_calculo.append(b)
        estado = ttk.Label(fila, text="")
        estado.pack(side="left", padx=8)
        return estado

    def _pestaña_caso(self, pestañas):
        marco = ttk.Frame(pestañas, padding=10)
        pestañas.add(marco, text="Tripartito")
        self.vars, self.suma = self._campos(marco, self.CAMPOS_CASO, self._revisar_suma)
        self.estado_caso = self._botones(marco, [
            ("Graficar anillo", lambda: self._iniciar_tri(correr_caso, "anillo")),
            ("Graficar matriz", lambda: self._iniciar_tri(correr_matriz, "matriz")),
            ("Graficar grafos", lambda: self._iniciar_tri(correr_grafos, "grafos")),
            ("Biparticiones asociadas", lambda: self._iniciar_tri(correr_asociadas,
                                                                  "biparticiones asociadas")),
        ])

    def _pestaña_bip(self, pestañas):
        marco = ttk.Frame(pestañas, padding=10)
        pestañas.add(marco, text="Bipartito")
        self.vars_bip, self.aviso_bip = self._campos(marco, self.CAMPOS_BIP, self._revisar_bip)
        self.estado_bip = self._botones(marco, [
            ("Graficar modos", lambda: self._iniciar_bip(correr_bip_modos, "modos")),
            ("Graficar matriz", lambda: self._iniciar_bip(correr_bip_matriz, "matriz")),
        ])
        self._revisar_bip()

    # --- pestaña "Barrido" ------------------------------------------------
    def _pestaña_barrido(self, pestañas):
        marco = ttk.Frame(pestañas, padding=10)
        pestañas.add(marco, text="Barrido")

        qué = ttk.LabelFrame(marco, text="Qué se barre", padding=10)
        qué.pack(fill="x")
        ttk.Label(qué, text="Barrer").grid(row=0, column=0, sticky="e", padx=(0, 4))
        self.b_param = tk.StringVar(value="h")
        combo = ttk.Combobox(qué, textvariable=self.b_param, values=["h", "γ", "partición", TODO],
                             state="readonly", width=18)
        combo.grid(row=0, column=1, sticky="w")
        combo.bind("<<ComboboxSelected>>", lambda *_: self._actualizar_fijos())
        ttk.Label(qué, text="valores").grid(row=0, column=2, sticky="e", padx=(16, 4))
        self.b_valores = tk.StringVar(value="0:1.5:0.1")
        self.b_entrada_valores = ttk.Entry(qué, textvariable=self.b_valores, width=40)
        self.b_entrada_valores.grid(row=0, column=3, sticky="we")
        qué.columnconfigure(3, weight=1)
        self.b_ayuda = ttk.Label(qué, text="", foreground="#5f5e5a")
        self.b_ayuda.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        fijos = ttk.LabelFrame(marco, text="Fijos", padding=10)
        fijos.pack(fill="x", pady=(8, 0))
        self.b_vars = {}
        self.b_entradas = {}
        campos = [("gamma", "γ", "0.5", 8), ("h", "h", "0.3", 8), ("N", "N", "200", 8),
                  ("particion", "partición", "66|66|68", 14)]
        for k, (clave, etiqueta, valor, ancho) in enumerate(campos):
            ttk.Label(fijos, text=etiqueta).grid(row=0, column=2 * k, sticky="e",
                                                 padx=(12 if k else 0, 4))
            v = tk.StringVar(value=valor)
            e = ttk.Entry(fijos, textvariable=v, width=ancho)
            e.grid(row=0, column=2 * k + 1, sticky="w")
            self.b_vars[clave] = v
            self.b_entradas[clave] = e

        opciones = ttk.LabelFrame(marco, text="Precisión y gráfica", padding=10)
        opciones.pack(fill="x", pady=(8, 0))
        ttk.Label(opciones, text="dps").grid(row=0, column=0, sticky="e", padx=(0, 4))
        self.b_dps = tk.StringVar(value="70")
        self.b_entrada_dps = ttk.Entry(opciones, textvariable=self.b_dps, width=6)
        self.b_entrada_dps.grid(row=0, column=1, sticky="w")
        self.b_critico = tk.BooleanVar(value=True)
        self.b_casilla_critico = ttk.Checkbutton(opciones, text="100 dígitos en h = ±1",
                                                 variable=self.b_critico)
        self.b_casilla_critico.grid(row=0, column=2, sticky="w", padx=(14, 0))
        ttk.Label(opciones, text="máx. modos").grid(row=0, column=3, sticky="e", padx=(18, 4))
        self.b_modos = tk.StringVar(value="30")
        ttk.Entry(opciones, textvariable=self.b_modos, width=6).grid(row=0, column=4, sticky="w")

        botones = ttk.Frame(marco)
        botones.pack(fill="x", pady=(10, 0))
        self.boton_revisar = ttk.Button(botones, text="Revisar barrido", command=self._revisar)
        self.boton_revisar.pack(side="left")
        self.boton_barrido = ttk.Button(botones, text="Correr barrido",
                                        command=self._iniciar_barrido)
        self.boton_barrido.pack(side="left", padx=8)
        self.boton_detener = ttk.Button(botones, text="Detener después del caso actual",
                                        command=self._pedir_detener, state="disabled")
        self.boton_detener.pack(side="left")

        progreso = ttk.Frame(marco)
        progreso.pack(fill="x", pady=(10, 0))
        self.barra = ttk.Progressbar(progreso, mode="determinate")
        self.barra.pack(fill="x")
        self.estado_barrido = ttk.Label(progreso, text="")
        self.estado_barrido.pack(anchor="w", pady=(4, 0))

    def _actualizar_fijos(self):
        p = self.b_param.get()
        todo = (p == TODO)
        barrido = {"h": "h", "γ": "gamma", "partición": "particion", TODO: None}[p]
        for clave, e in self.b_entradas.items():
            apagado = clave == barrido or (todo and clave in ("gamma", "h"))
            e.configure(state="disabled" if apagado else "normal")
        for e in (self.b_entrada_valores, self.b_entrada_dps, self.b_casilla_critico):
            e.configure(state="disabled" if todo else "normal")
        ayudas = {
            "h": "Rango inicio:fin:paso (p. ej. 0:1.5:0.1) o lista (p. ej. 0.2, 0.5, 0.9, 1.0).",
            "γ": "Rango inicio:fin:paso (p. ej. 0.25:1:0.25) o lista (p. ej. 0.25, 0.5, 1.0).",
            "partición": "Lista de particiones NA|NB|NC separadas por ; (p. ej. 66|66|68; 50|100|50).",
            TODO: "Usa todos los hamiltonianos guardados para este N, cada uno con su precisión, "
                  "y la partición fija. Al final: una gráfica con deslizador en h por cada γ.",
        }
        self.b_ayuda.configure(text=ayudas[p], wraplength=640)
        ejemplos = {"h": "0:1.5:0.1", "γ": "0.25:1:0.25", "partición": "66|66|68; 60|80|60; 50|100|50"}
        actual = self.b_valores.get().strip()
        if p in ejemplos and actual in ejemplos.values():
            self.b_valores.set(ejemplos[p])

    def _leer_barrido(self):
        p = self.b_param.get()
        v = {k: x.get().strip() for k, x in self.b_vars.items()}
        N = int(v["N"])
        max_modos = int(self.b_modos.get()) if self.b_modos.get().strip() else None
        if p == TODO:
            return p, casos_calculados(N, leer_particion(v["particion"])), max_modos
        gamma = None if p == "γ" else float(v["gamma"])
        h = None if p == "h" else float(v["h"])
        particion = None if p == "partición" else leer_particion(v["particion"])
        dps = int(self.b_dps.get())
        critico = 100 if self.b_critico.get() else None
        casos = armar_barrido(p, self.b_valores.get(), gamma, h, N, particion, dps, critico)
        max_modos = int(self.b_modos.get()) if self.b_modos.get().strip() else None
        return p, casos, max_modos

    # --- pestaña "Diagrama de fases" -------------------------------------
    CAMPOS_FASES = [  # (clave, etiqueta, valor inicial, ancho)
        ("N", "N", "200", 6),
        ("particion", "partición", "66|66|68", 12),
        ("gammas", "γ", "0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5", 34),
        ("hs", "h", "0, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5", 34),
        ("dps_nuevo", "dps para lo que falte", "150", 6),
        ("dps_min", "dps mínimo", "70", 6),
        ("dps_min_critico", "dps mínimo en líneas críticas", "100", 6),
        ("max_modos", "modos (cilindro)", "30", 6),
        ("k", "k (matriz)", "8", 6),
    ]

    def _pestaña_fases(self, pestañas):
        marco = ttk.Frame(pestañas, padding=10)
        pestañas.add(marco, text="Diagrama de fases")
        param = ttk.LabelFrame(marco, text="Malla y precisión", padding=8)
        param.pack(fill="x")
        self.f_vars = {}
        posiciones = [(0, 0), (0, 2), (0, 4), (1, 4), (2, 0), (2, 2), (2, 4), (1, 0), (1, 2)]
        for (clave, etiqueta, valor, ancho), (fila, col) in zip(self.CAMPOS_FASES, posiciones):
            ttk.Label(param, text=etiqueta).grid(row=fila, column=col, sticky="e", padx=(10, 4),
                                                 pady=3)
            v = tk.StringVar(value=valor)
            ttk.Entry(param, textvariable=v, width=ancho).grid(row=fila, column=col + 1,
                                                                sticky="w", pady=3)
            self.f_vars[clave] = v

        fila = ttk.Frame(marco)
        fila.pack(fill="x", pady=(8, 0))
        for texto, comando in (("Buscar", self._buscar_fases),
                               ("Calcular lo que falte y preparar fotos", self._iniciar_fases)):
            b = ttk.Button(fila, text=texto, command=comando)
            b.pack(side="left", padx=(0, 8))
            self.botones_calculo.append(b)
        self.boton_detener_fases = ttk.Button(fila, text="Detener después del caso actual",
                                              command=self._pedir_detener, state="disabled")
        self.boton_detener_fases.pack(side="left")
        self.estado_fases = ttk.Label(fila, text="", wraplength=640)
        self.estado_fases.pack(side="left", padx=10)

        paneles = ttk.Frame(marco)
        paneles.pack(fill="x", pady=(8, 0))
        ancho, alto = TAM_FOTO
        self.lienzo = tk.Canvas(paneles, width=ancho, height=alto, background="white",
                                highlightthickness=1, highlightbackground="#c3c2b7")
        self.lienzo.grid(row=0, column=0, padx=(0, 8))
        fotos = []
        for c in (1, 2):
            marco_foto = tk.Frame(paneles, width=ancho, height=alto, background="white",
                                  highlightthickness=1, highlightbackground="#c3c2b7")
            marco_foto.grid(row=0, column=c, padx=(0, 8))
            marco_foto.pack_propagate(False)
            foto = tk.Label(marco_foto, background="white", foreground="#5f5e5a", text="")
            foto.pack(fill="both", expand=True)
            fotos.append(foto)
        self.foto_cil, self.foto_mat = fotos
        self.info_fases = ttk.Label(paneles, text="Pasa el mouse por el diagrama.")
        self.info_fases.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.puntos = []
        self.punto_actual = None
        self.imagenes = {}                      # ruta -> tk.PhotoImage (para que no se borren)
        self.lienzo.bind("<Motion>", self._hover_fases)
        self.lienzo.bind("<Button-1>", self._clic_fases)

    def _leer_fases(self):
        v = {k: x.get().strip() for k, x in self.f_vars.items()}
        N = int(v["N"])
        NA, NB, NC = leer_particion(v["particion"])
        if NA + NB + NC != N:
            raise ValueError(f"La partición {NA}|{NB}|{NC} no suma N = {N}.")
        return dict(N=N, NA=NA, NB=NB, NC=NC, gammas=leer_numeros(v["gammas"]),
                    hs=leer_numeros(v["hs"]), dps_nuevo=int(v["dps_nuevo"]),
                    dps_min=int(v["dps_min"]), dps_min_critico=int(v["dps_min_critico"]),
                    max_modos=int(v["max_modos"]), k=int(v["k"]))

    def _buscar_fases(self, silencioso=False):
        try:
            p = self._leer_fases()
        except ValueError as e:
            self._escribir(f"Revisa el diagrama de fases: {e}")
            return None
        self.f_param = p
        self.puntos = puntos_diagrama(p["N"], p["NA"], p["NB"], p["NC"], p["gammas"], p["hs"],
                                      p["dps_nuevo"], p["dps_min"], p["dps_min_critico"])
        self.imagenes.clear()
        self.punto_actual = None
        self._dibujar_fases()
        faltan = [q for q in self.puntos if not q["listo"]]
        sin_foto = [q for q in self.puntos if q["listo"] and not all(
            os.path.exists(r) for r in self._rutas(q)[:2])]
        t = sum(tiempo_estimado(p["N"], *estado_caso(q["gamma"], q["h"], p["N"], p["NA"], p["NB"],
                                                     p["NC"], q["dps"])) * q["dps"] / 70.0
                for q in faltan) + 5.0 * (len(faltan) + len(sin_foto))
        resumen = (f"{len(self.puntos)} puntos: {len(self.puntos) - len(faltan)} calculados, "
                   f"{len(faltan)} por calcular, {len(sin_foto)} calculados sin foto. "
                   f"Tiempo aproximado para completar: {_formatear_tiempo(t)}.")
        self.estado_fases.configure(text=resumen)
        if not silencioso:
            self._escribir(f"\n=== Diagrama de fases: N={p['N']}, "
                           f"A|B|C = {p['NA']}|{p['NB']}|{p['NC']} ===\n   {resumen}")
            for q in faltan:
                self._escribir(f"   por calcular: γ={q['gamma']}, h={q['h']}  (dps{q['dps']})")
        return p

    def _rutas(self, q):
        p = self.f_param
        return rutas_caso(q["gamma"], q["h"], p["N"], p["NA"], p["NB"], p["NC"], q["dps"],
                          p["max_modos"], p["k"])

    # Coordenadas del lienzo
    def _ejes(self):
        ancho, alto = TAM_FOTO
        gmax = max([1.6] + [q["gamma"] + 0.1 for q in self.puntos])
        hmax = max([1.6] + [q["h"] + 0.1 for q in self.puntos])
        izq, der, arr, aba = 42, 12, 14, 34
        return (lambda g: izq + g / gmax * (ancho - izq - der),
                lambda h: alto - aba - h / hmax * (alto - arr - aba), gmax, hmax)

    def _dibujar_fases(self):
        c = self.lienzo
        c.delete("all")
        X, Y, gmax, hmax = self._ejes()
        gris, fuente = "#5f5e5a", ("TkDefaultFont", 9)
        c.create_line(X(0), Y(0), X(gmax), Y(0), fill=gris)
        c.create_line(X(0), Y(0), X(0), Y(hmax), fill=gris)
        for v in (0.5, 1.0, 1.5):
            if v <= gmax:
                c.create_line(X(v), Y(0), X(v), Y(0) + 4, fill=gris)
                c.create_text(X(v), Y(0) + 12, text=f"{v:g}", fill=gris, font=fuente)
            if v <= hmax:
                c.create_line(X(0) - 4, Y(v), X(0), Y(v), fill=gris)
                c.create_text(X(0) - 14, Y(v), text=f"{v:g}", fill=gris, font=fuente)
        c.create_text(X(gmax) - 6, Y(0) + 24, text="γ", font=("TkDefaultFont", 11, "italic"))
        c.create_text(X(0) - 28, Y(hmax) + 6, text="h", font=("TkDefaultFont", 11, "italic"))
        # Ising (gamma = 1), círculo gamma^2 + h^2 = 1, líneas críticas
        c.create_line(X(1), Y(0), X(1), Y(hmax), fill="#2f9e44", dash=(2, 3), width=2)
        c.create_text(X(1) + 4, Y(hmax) + 2, text="Ising", anchor="nw", font=fuente,
                      fill="#2f9e44")
        import math
        arco = []
        for j in range(61):
            t = j / 60 * math.pi / 2
            arco += [X(math.cos(t)), Y(math.sin(t))]
        c.create_line(*arco, fill="#1d3a6e", dash=(6, 4), width=2)
        c.create_line(X(0), Y(1), X(gmax), Y(1), fill="#c0262d", width=3)
        c.create_line(X(0), Y(0), X(0), Y(1), fill="#c0262d", width=3)
        for texto, g, h in (("Oscillatory", 0.35, 0.3), ("Ordered", min(gmax - 0.3, 1.3), 0.55),
                            ("Disordered", gmax * 0.62, (1 + hmax) / 2)):
            c.create_text(X(g), Y(h), text=texto, font=("TkDefaultFont", 10, "italic"),
                          fill="#383835")
        for q in self.puntos:
            x, y = X(q["gamma"]), Y(q["h"])
            if q["listo"]:
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#2a78d6", outline="white")
            else:
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="white", outline="#eb6834",
                              width=2)
        self.marca = c.create_oval(0, 0, 0, 0, outline="#141413", width=2, state="hidden")

    def _mas_cercano(self, evento):
        if not self.puntos:
            return None
        X, Y, gmax, hmax = self._ejes()
        return min(self.puntos, key=lambda q: (X(q["gamma"]) - evento.x) ** 2
                   + (Y(q["h"]) - evento.y) ** 2)

    def _imagen(self, ruta):
        if ruta not in self.imagenes:
            self.imagenes[ruta] = tk.PhotoImage(file=ruta) if os.path.exists(ruta) else None
        return self.imagenes[ruta]

    def _hover_fases(self, evento):
        q = self._mas_cercano(evento)
        if q is None or q is self.punto_actual:
            return
        self.punto_actual = q
        X, Y, _, _ = self._ejes()
        x, y = X(q["gamma"]), Y(q["h"])
        self.lienzo.coords(self.marca, x - 8, y - 8, x + 8, y + 8)
        self.lienzo.itemconfigure(self.marca, state="normal")
        foto_cil, foto_mat, _ = self._rutas(q)
        for etiqueta, ruta in ((self.foto_cil, foto_cil), (self.foto_mat, foto_mat)):
            img = self._imagen(ruta) if q["listo"] else None
            etiqueta.configure(image=img or "", text="" if img else (
                "Sin foto todavía:\nusa 'Calcular lo que falte y preparar fotos'." if q["listo"]
                else "Punto por calcular."))
        estado = "calculado" if q["listo"] else f"por calcular (con dps{q['dps']})"
        dps = f"dps{q['dps']}" + (" (exacto)" if q["dps"] < DPS_EXACTO else "")
        self.info_fases.configure(
            text=f"γ = {q['gamma']},  h = {q['h']}   ·   {dps}   ·   {estado}"
                 f"{'   ·   línea crítica' if es_critico(q['gamma'], q['h']) else ''}"
                 f"   ·   clic: abrir el cilindro 3D")

    def _clic_fases(self, evento):
        q = self._mas_cercano(evento)
        if q is None or self.ocupado:
            return
        if not q["listo"]:
            self._escribir(f"γ={q['gamma']}, h={q['h']} todavía no está calculado.")
            return
        p = self.f_param
        self._lanzar(correr_cilindro_punto,
                     dict(gamma=q["gamma"], h=q["h"], N=p["N"], NA=p["NA"], NB=p["NB"],
                          NC=p["NC"], dps=q["dps"], max_modos=p["max_modos"]),
                     self.estado_fases, f"\n=== cilindro 3D: γ={q['gamma']}, h={q['h']}, "
                                        f"dps={q['dps']} ===")

    def _iniciar_fases(self):
        p = self._buscar_fases()
        if p is None:
            return
        self.detener.clear()
        self._ocupar(True)
        self._escribir("Calculando los puntos que faltan y preparando las fotos...")
        threading.Thread(target=self._trabajo_fases, args=(p, list(self.puntos)),
                         daemon=True).start()

    def _trabajo_fases(self, p, puntos):
        t_inicio = time.time()
        fallidos = []
        try:
            faltan = [q for q in puntos if not q["listo"]]
            for j, q in enumerate(faltan):
                if self.detener.is_set():
                    break
                nombre = f"γ={q['gamma']}, h={q['h']}"
                self.cola.put(("estado_fases", f"Calculando {j + 1}/{len(faltan)}: {nombre} "
                                               f"(dps{q['dps']})"))
                self._log(f"[{j + 1}/{len(faltan)}] {nombre}  (dps{q['dps']})")
                try:
                    preparar_caso(q["gamma"], q["h"], p["N"], p["NA"], p["NB"], p["NC"],
                                  q["dps"], log=self._log)
                    q["listo"] = True
                except Exception as e:          # un caso con problemas no detiene el resto
                    fallidos.append(nombre)
                    self._log(f"   ERROR en {nombre}: {type(e).__name__}: {e}. Sigo.")
            listos = [q for q in puntos if q["listo"]]
            for j, q in enumerate(listos):
                if self.detener.is_set():
                    break
                rutas = rutas_caso(q["gamma"], q["h"], p["N"], p["NA"], p["NB"], p["NC"],
                                   q["dps"], p["max_modos"], p["k"])
                if all(os.path.exists(r) for r in rutas[:2]):
                    continue
                nombre = f"γ={q['gamma']}, h={q['h']}"
                self.cola.put(("estado_fases", f"Fotos {j + 1}/{len(listos)}: {nombre}"))
                try:
                    preparar_fotos(q["gamma"], q["h"], p["N"], p["NA"], p["NB"], p["NC"],
                                   q["dps"], p["max_modos"], p["k"])
                    self._log(f"   fotos listas: {nombre} (dps{q['dps']})")
                except Exception as e:
                    fallidos.append(f"fotos de {nombre}")
                    self._log(f"   ERROR en las fotos de {nombre}: {type(e).__name__}: {e}")
            mensaje = f"Diagrama de fases: terminado en {_formatear_tiempo(time.time() - t_inicio)}."
            if self.detener.is_set():
                mensaje = "Diagrama de fases: detenido. Al volver a correrlo, sigue donde quedó."
            if fallidos:
                mensaje += f" Fallaron {len(fallidos)}: {', '.join(fallidos)}."
            self.cola.put(("fin_fases", mensaje))
        except Exception as e:
            self.cola.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    # --- utilidades -------------------------------------------------------
    def _log(self, mensaje):
        self.cola.put(("log", mensaje))

    def _escribir(self, mensaje):
        self.texto.configure(state="normal")
        self.texto.insert("end", mensaje + "\n")
        self.texto.see("end")
        self.texto.configure(state="disabled")

    def _limpiar(self):
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.configure(state="disabled")

    def _revisar_suma(self):
        try:
            N, NA, NB, NC = (int(self.vars[k].get()) for k in ("N", "NA", "NB", "NC"))
        except ValueError:
            self.suma.configure(text="N, N_A, N_B y N_C deben ser enteros.")
            return
        s = NA + NB + NC
        if s == N:
            self.suma.configure(text=f"N_A + N_B + N_C = {s} = N  ✓")
        else:
            self.suma.configure(text=f"N_A + N_B + N_C = {s}, pero N = {N}  ✗")

    def _revisar_bip(self):
        try:
            N, NS = int(self.vars_bip["N"].get()), int(self.vars_bip["NS"].get())
        except ValueError:
            self.aviso_bip.configure(text="N y N_S deben ser enteros.")
            return
        if 1 <= NS <= N - 1:
            self.aviso_bip.configure(text=f"S = sitios 1..{NS},  T = sitios {NS + 1}..{N}  "
                                          f"({NS}|{N - NS})  ✓")
        else:
            self.aviso_bip.configure(text=f"N_S debe estar entre 1 y N - 1 = {N - 1}  ✗")

    def _ocupar(self, si):
        self.ocupado = si
        estado = "disabled" if si else "normal"
        for b in self.botones_calculo + [self.boton_revisar, self.boton_barrido]:
            b.configure(state=estado)
        for b in (self.boton_detener, self.boton_detener_fases):
            b.configure(state="normal" if si else "disabled")

    # --- tripartito y bipartito ------------------------------------------
    def _leer(self, variables, enteros):
        """Diccionario de parámetros, o None (y aviso en el registro) si hay un valor inválido."""
        v = {k: variables[k].get().strip() for k in variables}
        try:
            p = {k: (int(x) if k in enteros else float(x)) for k, x in v.items()
                 if k not in ("max_modos", "refinar", "k")}
            p["max_modos"] = int(v["max_modos"]) if v["max_modos"] else None
            p["refinar"] = int(v["refinar"]) if v["refinar"] else 1
            p["k"] = int(v["k"]) if v["k"] else 8
        except ValueError:
            self._escribir("Revisa los parámetros: hay un valor que no es un número.")
            return None
        return p

    def _iniciar_tri(self, tarea, que):
        p = self._leer(self.vars, ("N", "NA", "NB", "NC", "dps"))
        if p is None:
            return
        if p["NA"] + p["NB"] + p["NC"] != p["N"]:
            self._escribir("N_A + N_B + N_C tiene que ser igual a N.")
            return
        self._lanzar(tarea, p, self.estado_caso,
                     f"\n=== {que}: γ={p['gamma']}, h={p['h']}, N={p['N']}, "
                     f"A|B|C = {p['NA']}|{p['NB']}|{p['NC']}, dps={p['dps']} ===")

    def _iniciar_bip(self, tarea, que):
        p = self._leer(self.vars_bip, ("N", "NS", "dps"))
        if p is None:
            return
        if not 1 <= p["NS"] <= p["N"] - 1:
            self._escribir("N_S tiene que estar entre 1 y N - 1.")
            return
        self._lanzar(tarea, p, self.estado_bip,
                     f"\n=== {que}: γ={p['gamma']}, h={p['h']}, N={p['N']}, "
                     f"S|T = {p['NS']}|{p['N'] - p['NS']}, dps={p['dps']} ===")

    def _lanzar(self, tarea, p, estado, encabezado):
        self._ocupar(True)
        for b in (self.boton_detener, self.boton_detener_fases):
            b.configure(state="disabled")
        estado.configure(text="Calculando...")
        self._escribir(encabezado)
        threading.Thread(target=self._trabajo, args=(tarea, p), daemon=True).start()

    def _trabajo(self, tarea, p):
        t0 = time.time()
        try:
            htmls = tarea(log=self._log, **p)
            for html in htmls:
                webbrowser.open(Path(html).as_uri())
            n = len(htmls)
            self.cola.put(("fin", (htmls[-1] if htmls else None,
                                   f"Terminado en {time.time() - t0:.0f} s. Abriendo "
                                   f"{'la gráfica' if n == 1 else f'las {n} gráficas'} "
                                   "en el navegador.")))
        except Exception as e:
            self.cola.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    # --- barrido ----------------------------------------------------------
    def _revisar(self):
        try:
            p, casos, _ = self._leer_barrido()
        except ValueError as e:
            self._escribir(f"Revisa el barrido: {e}")
            return
        self._escribir(f"\n=== Revisión del barrido ({p}): {len(casos)} casos ===")
        total, nuevos = 0.0, 0
        for c in casos:
            ham, cov, diag = estado_caso(c["gamma"], c["h"], c["N"], c["NA"], c["NB"], c["NC"],
                                         c["dps"])
            t = tiempo_estimado(c["N"], ham, cov, diag)
            total += t
            if not diag:
                nuevos += 1
            pasos = ("listo" if diag else
                     "falta: local" if cov else
                     "falta: covarianza y local" if ham else "falta: todo")
            nombre = f"γ={c['gamma']}, {c['etiqueta']}" if p == TODO else c["etiqueta"]
            self._escribir(f"   {nombre:<22} dps{c['dps']:<4} {pasos:<28} ≈ "
                           f"{_formatear_tiempo(t)}")
        self._escribir(f"   {len(casos) - nuevos} ya calculados, {nuevos} por calcular. "
                       f"Tiempo total aproximado: {_formatear_tiempo(total)}.")

    def _iniciar_barrido(self):
        try:
            p, casos, max_modos = self._leer_barrido()
        except ValueError as e:
            self._escribir(f"Revisa el barrido: {e}")
            return
        self.detener.clear()
        self._ocupar(True)
        self.barra.configure(maximum=len(casos), value=0)
        self._escribir(f"\n=== Barrido ({p}): {len(casos)} casos ===")
        threading.Thread(target=self._trabajo_barrido, args=(p, casos, max_modos),
                         daemon=True).start()

    def _pedir_detener(self):
        self.detener.set()
        for b in (self.boton_detener, self.boton_detener_fases):
            b.configure(state="disabled")
        self._escribir("Me detengo al terminar el caso actual...")

    def _trabajo_barrido(self, p, casos, max_modos):
        t_inicio = time.time()
        try:
            estimados = [tiempo_estimado(c["N"], *estado_caso(c["gamma"], c["h"], c["N"], c["NA"],
                                                              c["NB"], c["NC"], c["dps"]))
                         for c in casos]
            # En "todo lo calculado" se arma una gráfica por cada gamma (deslizador en h).
            if p == TODO:
                grupos = {}
                for c in casos:
                    grupos.setdefault(c["gamma"], []).append(c)
                grupos = list(grupos.values())
            else:
                grupos = [casos]

            htmls, hechos_total, fallidos, k = [], 0, [], 0
            for grupo in grupos:
                listos, hechos = [], []
                for c in grupo:
                    if self.detener.is_set():
                        break
                    hecho = time.time() - t_inicio
                    previsto = sum(estimados[:k])
                    factor = hecho / previsto if previsto > 0 and k > 0 else 1.0
                    falta = factor * sum(estimados[k:])
                    nombre = f"γ={c['gamma']}, {c['etiqueta']}" if p == TODO else c["etiqueta"]
                    self.cola.put(("progreso", (k, len(casos),
                                                f"{k + 1}/{len(casos)}   caso actual: {nombre} "
                                                f"(dps{c['dps']})   faltan ≈ {_formatear_tiempo(falta)}")))
                    self._log(f"[{k + 1}/{len(casos)}] {nombre}  (dps{c['dps']})")
                    k += 1
                    try:
                        O = preparar_caso(c["gamma"], c["h"], c["N"], c["NA"], c["NB"], c["NC"],
                                          c["dps"], log=self._log)
                        listos.append((c["etiqueta"],
                                       regiones_para_grafica(O, c["N"], c["NA"], c["NB"], c["NC"],
                                                             max_modos)))
                        hechos.append(c)
                    except Exception as e:      # un caso con problemas no detiene el barrido
                        fallidos.append(nombre)
                        self._log(f"   ERROR en {nombre}: {type(e).__name__}: {e}. Sigo con el siguiente.")
                if listos:
                    htmls.append(self._grafica_barrido("h" if p == TODO else p, hechos, listos,
                                                       max_modos))
                    hechos_total += len(listos)
                if self.detener.is_set():
                    break

            if not htmls:
                self.cola.put(("fin_barrido", (None, "No se completó ningún caso.")))
                return
            for html in htmls:
                webbrowser.open(Path(html).as_uri())
            mensaje = (f"Barrido terminado: {hechos_total} casos en "
                       f"{_formatear_tiempo(time.time() - t_inicio)}, {len(htmls)} gráfica(s).")
            if hechos_total + len(fallidos) < len(casos):
                mensaje = (f"Barrido detenido: {hechos_total} de {len(casos)} casos. "
                           "Al volver a correrlo, reutiliza los que ya están.")
            if fallidos:
                mensaje += f" Fallaron {len(fallidos)}: {', '.join(fallidos)}."
            self.cola.put(("fin_barrido", (htmls[-1], mensaje)))
        except Exception as e:
            self.cola.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    def _grafica_barrido(self, p, hechos, listos, max_modos):
        """Arma y guarda la gráfica con deslizador de un grupo de casos ya calculados."""
        self._log(f"Armando la gráfica con deslizador ({len(listos)} casos)...")
        graf = _modulo("graf")
        salida = os.path.join(PROGRAMAS, "Resultados_modos")
        os.makedirs(salida, exist_ok=True)
        html = os.path.join(salida, nombre_barrido(p, hechos))
        c0 = hechos[0]
        fijos = {"h": f"γ={c0['gamma']}, A|B|C = {c0['NA']}|{c0['NB']}|{c0['NC']}",
                 "γ": f"h={c0['h']}, A|B|C = {c0['NA']}|{c0['NB']}|{c0['NC']}",
                 "partición": f"γ={c0['gamma']}, h={c0['h']}"}[p]
        graf.graficar_cilindro_deslizador(
            listos, c0["N"], nombre_parametro=p, max_modos=max_modos, archivo=html,
            titulo=f"Barrido en {p}   ({fijos}, N={c0['N']})")
        self._log(f"Guardada en Resultados_modos/{os.path.basename(html)}")
        return html

    # --- cola de mensajes -------------------------------------------------
    def _leer_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "log":
                    self._escribir(dato)
                elif tipo == "progreso":
                    k, n, texto = dato
                    self.barra.configure(value=k)
                    self.estado_barrido.configure(text=texto)
                elif tipo == "estado_fases":
                    self.estado_fases.configure(text=dato)
                elif tipo == "fin_fases":
                    self._escribir(dato)
                    self._ocupar(False)
                    self._buscar_fases(silencioso=True)
                    self.estado_fases.configure(text=dato)
                elif tipo in ("fin", "fin_barrido"):
                    html, mensaje = dato
                    if html:
                        self.ultimo_html = html
                        self.boton_abrir.configure(state="normal")
                    self._escribir(mensaje)
                    self.estado_caso.configure(text="")
                    self.estado_bip.configure(text="")
                    self.estado_fases.configure(text="")
                    if tipo == "fin_barrido":
                        self.barra.configure(value=self.barra["maximum"])
                        self.estado_barrido.configure(text=mensaje)
                    self._ocupar(False)
                elif tipo == "error":
                    self._escribir("ERROR  " + dato)
                    self.estado_caso.configure(text="")
                    self.estado_bip.configure(text="")
                    self.estado_fases.configure(text="Error: ver el registro.")
                    self.estado_barrido.configure(text="Error: ver el registro.")
                    self._ocupar(False)
        except queue.Empty:
            pass
        self.after(100, self._leer_cola)

    def _abrir_ultima(self):
        if self.ultimo_html and os.path.exists(self.ultimo_html):
            webbrowser.open(Path(self.ultimo_html).as_uri())


if __name__ == "__main__":
    Consola().mainloop()