"""
Visualización de matrices de covarianza.

Funciona con cualquier número de regiones (tripartición A | B | C o bipartición S | T);
los grafos (graficar_grafos) son solo para la tripartición.

Matriz reducida por zoom de la covarianza diagonalizada localmente, M_tilde = O M O^T con
O = O_A (+) O_B (+) O_C. Como en M_tilde los bloques a'a' y b'b' son cero, toda la
información está en
    W_tilde[mu, nu] = <-i a'_mu b'_nu>,
con filas = todos los a' (regiones A, B, C) y columnas = todos los b'. Sus bloques
diagonales son -diag(lambda) de cada región y los de fuera, los acoplamientos entre regiones.
Los modos de cada región están ordenados del más entrelazado al menos, así que basta mirar
los primeros k de cada una: W_tilde reducida, de 3k x 3k.

Requiere: numpy, plotly   (pip install numpy plotly)
"""
import os
import numpy as np
import plotly.graph_objects as go

PROGRAMAS = os.path.dirname(os.path.abspath(__file__))


def W_tilde(M_tilde, *tamaños):
    """Bloque a'-b' de M_tilde, en el orden (a'_R1 | a'_R2 | ...) x (b'_R1 | b'_R2 | ...).
    M_tilde está en el orden (a'_R1 | b'_R1 | a'_R2 | b'_R2 | ...).
    tamaños: número de sitios de cada región, p. ej. W_tilde(M_tilde, NA, NB, NC) o
    W_tilde(M_tilde, NS, NT)."""
    filas, cols, inicio = [], [], 0
    for n in tamaños:
        filas += list(range(inicio, inicio + n))
        cols += list(range(inicio + n, inicio + 2 * n))
        inicio += 2 * n
    return M_tilde[np.ix_(filas, cols)]


def reducir_regiones(W, tamaños, k):
    """Los primeros k modos de cada región (filas y columnas): matriz de (n k) x (n k),
    con n = número de regiones."""
    k = min([k] + list(tamaños))
    indices, inicio = [], 0
    for n in tamaños:
        indices += list(range(inicio, inicio + k))
        inicio += n
    return W[np.ix_(indices, indices)], k


def reducir(W, NA, NB, NC, k):
    """Los primeros k modos de cada región (filas y columnas): matriz de 3k x 3k."""
    return reducir_regiones(W, (NA, NB, NC), k)


def graficar_reducida(Wk, k, lams=None, titulo="", archivo=None, piso=1e-20, etiquetas="ABC"):
    """Mapa de calor de la matriz reducida en escala log (ver figura_reducida): lo guarda en
    'archivo' (.html) o, si no se da, lo muestra."""
    fig = figura_reducida(Wk, k, lams, titulo, piso, etiquetas)
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def figura_reducida(Wk, k, lams=None, titulo="", piso=1e-20, etiquetas="ABC"):
    """
    Figura (plotly) del mapa de calor de la matriz reducida. Color: log10 |W|; al pasar el mouse se ve el valor
    con signo. Las líneas separan las regiones A, B, C.
    lams: opcional, (lam_A, lam_B, lam_C) para rotular las filas con su lambda.
    etiquetas: nombres de las regiones, p. ej. "ABC" o ("A", "BC").
    piso: los valores con |W| < piso se muestran como piso.
    """
    nombres = [f"{r}{m}" for r in etiquetas for m in range(k)]
    if lams is not None:
        filas = [f"{r}{m}  (λ={lams[j][m]:.3g})" for j, r in enumerate(etiquetas)
                 for m in range(k)]
    else:
        filas = nombres
    n = len(etiquetas) * k
    log = np.log10(np.maximum(np.abs(Wk), piso))
    texto = [[f"a'({filas[i].split()[0]}) – b'({nombres[j]})<br>W = {Wk[i, j]:.6g}"
              for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=log, x=nombres, y=filas, text=texto, hoverinfo="text",
        colorscale="Blues", zmin=np.log10(piso), zmax=0,
        colorbar=dict(title=dict(text="log₁₀|W|", side="right"), thickness=15)))
    for p in [j * k - 0.5 for j in range(1, len(etiquetas))]:
        fig.add_shape(type="line", x0=p, x1=p, y0=-0.5, y1=n - 0.5,
                      line=dict(color="#383835", width=1.5))
        fig.add_shape(type="line", y0=p, y1=p, x0=-0.5, x1=n - 0.5,
                      line=dict(color="#383835", width=1.5))
    fig.update_layout(
        title=dict(text=titulo, x=0.5),
        xaxis=dict(title="b' (modo local)", side="top", tickangle=0),
        yaxis=dict(title="a' (modo local)", autorange="reversed"),
        width=900, height=880, margin=dict(l=150, r=40, t=140, b=40))
    return fig


def graficar_reducida_lineal(Wk, k, lams=None, titulo="", archivo=None, etiquetas="ABC"):
    """
    Mapa de calor de la matriz reducida en escala lineal con signo:
    rojo = negativo, blanco = 0, azul = positivo (de -1 a 1).
    Muestra claramente las parejas (±1) y la diagonal (-lambda); los acoplamientos
    pequeños casi no se ven (para eso, graficar_reducida en escala log).
    lams: opcional, (lam_A, lam_B, lam_C) para rotular las filas con su lambda.
    etiquetas: nombres de las regiones, p. ej. "ABC" o ("A", "BC").
    """
    nombres = [f"{r}{m}" for r in etiquetas for m in range(k)]
    if lams is not None:
        filas = [f"{r}{m}  (λ={lams[j][m]:.3g})" for j, r in enumerate(etiquetas)
                 for m in range(k)]
    else:
        filas = nombres
    n = len(etiquetas) * k
    texto = [[f"a'({filas[i].split()[0]}) – b'({nombres[j]})<br>W = {Wk[i, j]:.6g}"
              for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=Wk, x=nombres, y=filas, text=texto, hoverinfo="text",
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="W", side="right"), thickness=15)))
    for p in [j * k - 0.5 for j in range(1, len(etiquetas))]:
        fig.add_shape(type="line", x0=p, x1=p, y0=-0.5, y1=n - 0.5,
                      line=dict(color="#383835", width=1.5))
        fig.add_shape(type="line", y0=p, y1=p, x0=-0.5, x1=n - 0.5,
                      line=dict(color="#383835", width=1.5))
    fig.update_layout(
        title=dict(text=titulo, x=0.5),
        xaxis=dict(title="b' (modo local)", side="top", tickangle=0),
        yaxis=dict(title="a' (modo local)", autorange="reversed"),
        width=900, height=880, margin=dict(l=150, r=40, t=140, b=40))
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


COLORES = ["#2a78d6", "#eb6834", "#2f9e44"]      # regiones A, B, C


def agrupar_modos(lams, NA, NB, lam_max=0.9999, tol=1e-6):
    """
    Nodos del grafo: los modos con lambda < lam_max de cada región. Los modos de una misma
    región con lambda iguales (|diferencia| < tol) se juntan en un solo nodo, porque su base
    es arbitraria. Devuelve una lista de (región 0/1/2, índices globales en W_tilde, lambda).
    """
    inicios = (0, NA, NA + NB)
    nodos = []
    for r, lam in enumerate(lams):
        grupo = []
        for m, l in enumerate(lam):
            if l >= lam_max:
                break                             # están ordenados de menor a mayor
            if grupo and abs(l - lam[grupo[-1]]) < tol:
                grupo.append(m)
            else:
                if grupo:
                    nodos.append((r, [inicios[r] + i for i in grupo], lam[grupo[0]]))
                grupo = [m]
        if grupo:
            nodos.append((r, [inicios[r] + i for i in grupo], lam[grupo[0]]))
    return nodos


def _nombre_nodo(r, indices, NA, NB):
    inicio = (0, NA, NA + NB)[r]
    return "ABC"[r] + ",".join(str(i - inicio) for i in indices)


def _entropia(l):
    p = (1 + l) / 2
    return 0.0 if p >= 1 else -(p * np.log(p) + (1 - p) * np.log(1 - p))


def graficar_grafos(W, lams, NA, NB, NC, titulo="", archivo=None, lam_max=0.9999,
                    umbral=1e-3, tol=1e-6):
    """
    Dos grafos de acoplamientos entre regiones, lado a lado, a partir de W_tilde completa (N x N).
      Izquierda, nodos = modos: arista entre dos modos de regiones distintas con peso
          p = (W[mu, nu]^2 + W[nu, mu]^2) / 2   (suma de las dos direcciones a'-b').
          Por pureza, la suma de p de un modo es 1 - lambda^2. Tamaño del nodo: su entropía.
      Derecha, nodos = Majoranas (a' círculo, b' rombo): arista a'_mu - b'_nu entre regiones
          distintas con peso W[mu, nu]^2; línea gris punteada a'_m - b'_m dentro del modo con
          peso lambda^2. La suma de pesos de cada Majorana es 1.
    Modos degenerados agrupados (ver agrupar_modos); los pesos de un grupo se suman.
    Solo se dibujan las aristas con peso > umbral.
    """
    from plotly.subplots import make_subplots

    nodos = agrupar_modos(lams, NA, NB, lam_max, tol)
    nombres = [_nombre_nodo(r, ind, NA, NB) for r, ind, _ in nodos]

    # Posiciones: cada región en una esquina del triángulo (orden A -> B -> C antihorario),
    # sus modos apilados hacia afuera; en la vista de Majoranas, b' hacia la región siguiente
    # y a' hacia la anterior.
    angulos = np.radians([90, 210, 330])
    pos_modo, pos_a, pos_b, cuenta = {}, {}, {}, [0, 0, 0]
    for i, (r, _, _) in enumerate(nodos):
        c = np.array([np.cos(angulos[r]), np.sin(angulos[r])])
        t = np.array([-c[1], c[0]])                        # tangente antihoraria
        centro = (1.0 + 0.45 * cuenta[r]) * c
        cuenta[r] += 1
        pos_modo[i] = centro
        pos_a[i] = centro - 0.22 * t
        pos_b[i] = centro + 0.22 * t

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Nodos = modos", "Nodos = Majoranas"),
                        horizontal_spacing=0.04)
    ancho = lambda w: 1 + 12 * min(w, 1.0)

    def arista(p0, p1, w, col, texto, color="rgba(42,120,214,0.55)", punteada=False):
        fig.add_trace(go.Scatter(x=[p0[0], p1[0]], y=[p0[1], p1[1]], mode="lines",
                                 line=dict(width=ancho(w), color=color,
                                           dash="dot" if punteada else "solid"),
                                 hoverinfo="text", text=texto, showlegend=False),
                      row=1, col=col)

    # Aristas
    for i, (ri, gi, li) in enumerate(nodos):
        for j, (rj, gj, lj) in enumerate(nodos):
            if ri == rj:
                continue
            q = float((W[np.ix_(gi, gj)] ** 2).sum())           # a'_i con b'_j
            if q > umbral:
                arista(pos_a[i], pos_b[j], q, 2, f"a'({nombres[i]}) – b'({nombres[j]}): {q:.3g}")
            if j > i:
                p = float(((W[np.ix_(gi, gj)] ** 2).sum() + (W[np.ix_(gj, gi)] ** 2).sum()) / 2)
                if p > umbral:
                    arista(pos_modo[i], pos_modo[j], p, 1, f"{nombres[i]} – {nombres[j]}: {p:.3g}")
        interno = float((W[np.ix_(gi, gi)] ** 2).sum())          # a'_m - b'_m: lambda^2
        if interno > umbral:
            arista(pos_a[i], pos_b[i], interno, 2, f"dentro de {nombres[i]}: λ² = {interno:.3g}",
                   color="rgba(120,120,115,0.6)", punteada=True)

    # Nodos
    colores = [COLORES[r] for r, _, _ in nodos]
    hover = [f"{n}: λ = {l:.6g}" for n, (_, _, l) in zip(nombres, nodos)]
    tamaños = [14 + 36 * _entropia(l) / np.log(2) for _, _, l in nodos]
    fig.add_trace(go.Scatter(
        x=[pos_modo[i][0] for i in range(len(nodos))], y=[pos_modo[i][1] for i in range(len(nodos))],
        mode="markers+text", text=nombres, textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(size=tamaños, color=colores, line=dict(color="white", width=1.5)),
        showlegend=False), row=1, col=1)
    for pos, simbolo, prefijo in ((pos_a, "circle", "a'"), (pos_b, "diamond", "b'")):
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in range(len(nodos))], y=[pos[i][1] for i in range(len(nodos))],
            mode="markers+text", text=[f"{prefijo}{n}" for n in nombres],
            textposition="top center", hoverinfo="text",
            marker=dict(size=16, symbol=simbolo, color=colores, line=dict(color="white", width=1.5)),
            showlegend=False), row=1, col=2)

    rango = 1.0 + 0.45 * max(cuenta) + 0.4
    for col in (1, 2):
        fig.update_xaxes(visible=False, range=[-rango, rango], row=1, col=col)
        fig.update_yaxes(visible=False, range=[-rango * 0.75, rango], row=1, col=col,
                         scaleanchor="x" if col == 1 else "x2")
    fig.update_layout(title=dict(text=titulo, x=0.5), width=1400, height=720,
                      plot_bgcolor="white", margin=dict(l=20, r=20, t=90, b=20))
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def _leer(prefijo, etiquetas, dps):
    """M_tilde y los lambda de una partición guardada en Resultados_diag_local/dps<n>
    (en doble precisión, suficiente para graficar)."""
    base = os.path.join(PROGRAMAS, "Resultados_diag_local")
    carpeta = os.path.join(base, f"dps{dps}")
    archivo = os.path.join(carpeta, prefijo + "_Mtilde.txt")
    if not os.path.exists(archivo):
        otros = [d for d in sorted(os.listdir(base))
                 if os.path.exists(os.path.join(base, d, prefijo + "_Mtilde.txt"))]
        pista = f" Sí está en: {', '.join(otros)}." if otros else ""
        raise FileNotFoundError(f"No encontré {prefijo}_Mtilde.txt en {carpeta}.{pista}")
    M_tilde = np.loadtxt(archivo)
    lams = [np.loadtxt(os.path.join(carpeta, f"{prefijo}_lambda_{r}.txt"), ndmin=1)
            for r in etiquetas]
    return M_tilde, lams


def leer_caso(gamma, h, N, NA, NB, dps):
    """M_tilde y (lam_A, lam_B, lam_C) de la tripartición A | B | C."""
    NC = N - NA - NB
    prefijo = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}_A{NA}_B{NB}_C{NC}"
    M_tilde, lams = _leer(prefijo, "ABC", dps)
    return M_tilde, lams, prefijo


def leer_biparticion(gamma, h, N, NS, inicio, dps):
    """M_tilde y (lam_S, lam_T) de la bipartición S | T (S = arco de NS sitios desde 'inicio',
    T = el resto; ver diag_local)."""
    prefijo = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}_bip_ini{inicio % N}_NS{NS}"
    M_tilde, lams = _leer(prefijo, "ST", dps)
    return M_tilde, lams, prefijo


def _salida():
    salida = os.path.join(PROGRAMAS, "Resultados_matrices")
    os.makedirs(salida, exist_ok=True)
    return salida


def graficar_caso(gamma, h, N, NA, NB, dps, k=8):
    """Matriz W_tilde reducida (primeros k modos por región) de la tripartición A | B | C,
    en escala log y en escala lineal. Guarda en Resultados_matrices:
        <prefijo>_dps<dps>_W_reducida_k<k>.html,  <prefijo>_dps<dps>_W_reducida_lineal_k<k>.html
    Devuelve la lista de archivos."""
    NC = N - NA - NB
    M_tilde, lams, prefijo = leer_caso(gamma, h, N, NA, NB, dps)
    Wk, k = reducir(W_tilde(M_tilde, NA, NB, NC), NA, NB, NC, k)
    datos = f"γ={gamma}, h={h}, N={N}, A|B|C = {NA}|{NB}|{NC}, dps={dps}, {k} modos por región"
    archivos = [os.path.join(_salida(), f"{prefijo}_dps{dps}_W_reducida_k{k}.html"),
                os.path.join(_salida(), f"{prefijo}_dps{dps}_W_reducida_lineal_k{k}.html")]
    graficar_reducida(Wk, k, lams, archivo=archivos[0], titulo=f"W̃ reducida<br>({datos})")
    graficar_reducida_lineal(Wk, k, lams, archivo=archivos[1],
                             titulo=f"W̃ reducida, escala lineal<br>({datos})")
    for a in archivos:
        print("Guardada en", a, flush=True)
    return archivos


def graficar_caso_grafos(gamma, h, N, NA, NB, dps, lam_max=0.9999, umbral=1e-3, tol=1e-6):
    """Grafos de acoplamientos (nodos = modos y nodos = Majoranas) de la tripartición A | B | C.
    Guarda en Resultados_matrices/<prefijo>_dps<dps>_grafos.html. Devuelve [archivo]."""
    NC = N - NA - NB
    M_tilde, lams, prefijo = leer_caso(gamma, h, N, NA, NB, dps)
    archivo = os.path.join(_salida(), f"{prefijo}_dps{dps}_grafos.html")
    graficar_grafos(W_tilde(M_tilde, NA, NB, NC), lams, NA, NB, NC, archivo=archivo,
                    lam_max=lam_max, umbral=umbral, tol=tol,
                    titulo=f"Acoplamientos entre regiones (γ={gamma}, h={h}, N={N}, "
                           f"A|B|C = {NA}|{NB}|{NC}, dps={dps})")
    print("Guardada en", archivo, flush=True)
    return [archivo]


def graficar_caso_biparticion(gamma, h, N, NS, inicio=0, dps=70, k=8, etiquetas=("S", "T")):
    """Matriz W_tilde reducida (primeros k modos por región) de la bipartición S | T, en escala
    log y en escala lineal. Guarda en Resultados_matrices:
        <prefijo>_dps<dps>_W_reducida_k<k>.html,  <prefijo>_dps<dps>_W_reducida_lineal_k<k>.html
    con <prefijo> = anillo_par_N{N}_gamma{gamma}_h{h}_bip_ini{inicio}_NS{NS}.
    etiquetas: solo cambian los nombres de S y T en las gráficas (p. ej. ("A", "BC")).
    Devuelve la lista de archivos."""
    NT = N - NS
    M_tilde, lams, prefijo = leer_biparticion(gamma, h, N, NS, inicio, dps)
    Wk, k = reducir_regiones(W_tilde(M_tilde, NS, NT), (NS, NT), k)
    nombre = f"{etiquetas[0]} | {etiquetas[1]}"
    datos = (f"γ={gamma}, h={h}, N={N}, {nombre} = {NS}|{NT}, {etiquetas[0]} desde el sitio "
             f"{inicio % N + 1}, dps={dps}, {k} modos por región")
    archivos = [os.path.join(_salida(), f"{prefijo}_dps{dps}_W_reducida_k{k}.html"),
                os.path.join(_salida(), f"{prefijo}_dps{dps}_W_reducida_lineal_k{k}.html")]
    graficar_reducida(Wk, k, lams, archivo=archivos[0], etiquetas=etiquetas,
                      titulo=f"W̃ reducida, bipartición {nombre}<br>({datos})")
    graficar_reducida_lineal(Wk, k, lams, archivo=archivos[1], etiquetas=etiquetas,
                             titulo=f"W̃ reducida, escala lineal, bipartición {nombre}<br>({datos})")
    for a in archivos:
        print("Guardada en", a, flush=True)
    return archivos


if __name__ == "__main__":
    gamma = 0.5
    h = 0.3
    N = 200
    NA, NB = 66, 66          # C es el resto: N - NA - NB
    dps = 70
    k = 8                    # modos por región en la matriz reducida

    graficar_caso(gamma, h, N, NA, NB, dps, k)
    graficar_caso_grafos(gamma, h, N, NA, NB, dps)
    # Bipartición S = [0, NS), T = el resto:
    # graficar_caso_biparticion(gamma, h, N, NS=100, dps=dps, k=k)