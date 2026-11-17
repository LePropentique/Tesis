"""
Funciones de modo a partir de una ortogonal O (global o local).

O actúa sobre Majoranas en el orden (a_1..a_n | b_1..b_n), gamma' = O gamma, y el modo
físico m está formado por los Majoranas emparejados
    gamma'_alpha = a'_m  (fila alpha = m)      y      gamma'_beta = b'_m  (fila beta = n + m).

Función de modo del modo físico m en el sitio i, con
    u_i = F(gamma'_alpha)_i = O[alpha, a_i]   (a'_m sobre a_i, columna i)
    v_i = F(gamma'_beta)_i  = O[beta,  b_i]   (b'_m sobre b_i, columna n + i):
    altura  H_m(i) = u_i^2 + v_i^2
    color   C_m(i) = cos(2 theta) = (u_i^2 - v_i^2) / (u_i^2 + v_i^2),   theta = arctan(v_i / u_i)
C = +1: todo a,  C = -1: todo b,  C = 0: mitad y mitad. No cambia si (u, v) -> (-u, -v).

Requiere: mpmath, plotly   (pip install mpmath plotly)
"""
from mpmath import mp


def funcion_de_modos(O):
    """
    Recibe una ortogonal O de 2n x 2n (orden (a | b)) y devuelve dos listas n x n (floats):
        H[m][i] = u_i^2 + v_i^2                          (altura)
        C[m][i] = (u_i^2 - v_i^2) / (u_i^2 + v_i^2)      (cos 2 theta; 0 si u = v = 0)
    con fila m = modo físico m y columna i = sitio i.
    """
    n = O.rows // 2
    H = [[0.0] * n for _ in range(n)]
    C = [[0.0] * n for _ in range(n)]
    for m in range(n):
        alpha, beta = m, n + m
        for i in range(n):
            u2 = O[alpha, i] ** 2
            v2 = O[beta, n + i] ** 2
            H[m][i] = float(u2 + v2)
            C[m][i] = float((u2 - v2) / (u2 + v2)) if u2 + v2 != 0 else 0.0
    return H, C


# Versión con cuadrados (peso del modo en el sitio, sin signo; cumple sum_i F_m(i) = 1).
# Para usarla, descomentarla y comentar la de arriba.
#
# def funcion_de_modos(O):
#     n = O.rows // 2
#     F = mp.matrix(n, n)
#     for m in range(n):
#         alpha, beta = m, n + m
#         for i in range(n):
#             F[m, i] = (O[alpha, i] ** 2 + O[alpha, n + i] ** 2
#                        + O[beta, i] ** 2 + O[beta, n + i] ** 2) / 2
#     return F


# ---------------------------------------------------------------------------
# Gráfica 3D (plotly) como superficie de una región: x = sitio i, y = modo m, z = F_m(i)
# Se guarda como .html: se abre en el navegador y se puede rotar y hacer zoom.
# ---------------------------------------------------------------------------
import os
import plotly.graph_objects as go


def refinar_malla(H, C, factor):
    """Interpolación visual (bicúbica) entre sitios y modos, solo para que la superficie se vea
    suave: no agrega información."""
    import numpy as np
    from scipy.ndimage import zoom
    Hf = np.clip(zoom(np.array(H), factor, order=3), 0.0, None)
    Cf = np.clip(zoom(np.array(C), factor, order=3), -1.0, 1.0)
    return Hf, Cf


def graficar_funcion_de_modos(H, C, sitios=None, titulo="", archivo=None, max_modos=None,
                              refinar=1):
    """
    H, C: altura y cos 2 theta de funcion_de_modos (fila m = modo, columna i = sitio).
    La altura de la superficie es H y el color es C: azul = todo a, rojo = todo b.
    sitios: números de sitio de la región (por defecto 1..n).
    max_modos: dibujar solo los primeros max_modos modos (None = todos).
    archivo: ruta del .html a guardar; si es None, abre la gráfica en el navegador.
    refinar: factor de interpolación visual (1 = datos tal cual; 4 = 4 veces más fino).
    """
    import numpy as np
    n = len(H)
    sitios = list(sitios) if sitios is not None else list(range(1, n + 1))
    k = n if max_modos is None else min(n, max_modos)
    Z = np.array(H[:k])
    S = np.array(C[:k])
    xs = np.array(sitios, dtype=float)
    ys = np.arange(k, dtype=float)
    if refinar > 1:
        Z, S = refinar_malla(Z, S, refinar)
        xs = np.linspace(xs[0], xs[-1], Z.shape[1])
        ys = np.linspace(ys[0], ys[-1], Z.shape[0])

    fig = go.Figure(go.Surface(
        x=xs, y=ys, z=Z, surfacecolor=S, cmin=-1.0, cmax=1.0,
        colorscale=[[0.0, "#c0392b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]],
        colorbar=dict(title="cos 2θ", thickness=15, len=0.6, tickvals=[-1, 0, 1],
                      ticktext=["todo b", "mitad y mitad", "todo a"]),
        customdata=S,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1, roughness=0.7),
        lightposition=dict(x=100, y=200, z=1000),
        hovertemplate="sitio %{x:.1f}<br>modo %{y:.1f}<br>u²+v² = %{z:.4g}"
                      "<br>cos 2θ = %{customdata:.3f}<extra></extra>"))
    fig.update_layout(
        title=dict(text=titulo, x=0.5),
        scene=dict(xaxis_title="sitio i", yaxis_title="modo m", zaxis_title="u² + v²",
                   aspectmode="manual", aspectratio=dict(x=1.4, y=1.0, z=0.6),
                   camera=dict(eye=dict(x=-1.5, y=-1.6, z=1.0))),
        margin=dict(l=0, r=0, t=50, b=0), width=1000, height=750)

    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def graficar_regiones_juntas(regiones, titulo="", archivo=None, max_modos=None, refinar=1):
    """
    Las tres (o las que sean) regiones en una misma gráfica, cada una como superficie sobre
    sus propios sitios de la cadena, con la misma escala de color (cos 2 theta).
    regiones: lista de (nombre, H, C, sitios), con H y C de funcion_de_modos.
    """
    import numpy as np
    trazas = []
    kmax = 0
    for r, (nombre, H, C, sitios) in enumerate(regiones):
        n = len(H)
        k = n if max_modos is None else min(n, max_modos)
        kmax = max(kmax, k)
        Z = np.array(H[:k])
        S = np.array(C[:k])
        xs = np.array(list(sitios), dtype=float)
        ys = np.arange(k, dtype=float)
        if refinar > 1:
            Z, S = refinar_malla(Z, S, refinar)
            xs = np.linspace(xs[0], xs[-1], Z.shape[1])
            ys = np.linspace(ys[0], ys[-1], Z.shape[0])
        trazas.append(go.Surface(
            x=xs, y=ys, z=Z, surfacecolor=S, cmin=-1.0, cmax=1.0, name=f"Región {nombre}",
            colorscale=[[0.0, "#c0392b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]],
            showscale=(r == 0),
            colorbar=dict(title="cos 2θ", thickness=15, len=0.6, tickvals=[-1, 0, 1],
                          ticktext=["todo b", "mitad y mitad", "todo a"]),
            customdata=S,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1, roughness=0.7),
            lightposition=dict(x=100, y=200, z=1000),
            hovertemplate=f"Región {nombre}<br>" + "sitio %{x:.1f}<br>modo %{y:.1f}"
                          "<br>u²+v² = %{z:.4g}<br>cos 2θ = %{customdata:.3f}<extra></extra>"))

    fig = go.Figure(trazas)
    fig.update_layout(
        title=dict(text=titulo, x=0.5),
        scene=dict(xaxis_title="sitio i", yaxis_title="modo m", zaxis_title="u² + v²",
                   aspectmode="manual", aspectratio=dict(x=2.2, y=1.0, z=0.6),
                   camera=dict(eye=dict(x=-1.0, y=-2.3, z=1.2))),
        margin=dict(l=0, r=0, t=50, b=0), width=1300, height=750)
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def trazas_cilindro(regiones, N, max_modos=None, refinar=1):
    """
    Trazas de plotly de las regiones sobre el anillo en coordenadas cilíndricas:
        ángulo  phi = 2 pi (i - 1/2) / N      (sitio i)
        radio   r   = R_m, el modo más entrelazado (m = 0) afuera y los demás hacia adentro
        altura  z   = u^2 + v^2
    con color cos 2 theta. Cada región es una superficie sobre su sector del anillo.
    regiones: lista de (nombre, H, C, sitios), con H y C de funcion_de_modos.
    Devuelve (trazas, zmax). Siempre son 3 trazas por región: superficie, corte y nombre.
    """
    import numpy as np
    k_reg = [len(H) if max_modos is None else min(len(H), max_modos) for _, H, _, _ in regiones]
    kmax = max(k_reg)
    R_ext, R_int = 1.0, 0.35                          # radios del modo 0 y del último modo
    paso = (R_ext - R_int) / max(kmax - 1, 1)

    trazas = []
    zmax = 0.0
    for r, ((nombre, H, C, sitios), k) in enumerate(zip(regiones, k_reg)):
        Z = np.array(H[:k])
        S = np.array(C[:k])
        sit = np.array(list(sitios), dtype=float)
        mod = np.arange(k, dtype=float)
        if refinar > 1:
            Z, S = refinar_malla(Z, S, refinar)
            sit = np.linspace(sit[0], sit[-1], Z.shape[1])
            mod = np.linspace(mod[0], mod[-1], Z.shape[0])
        zmax = max(zmax, Z.max())
        phi = 2 * np.pi * (sit - 0.5) / N
        rad = R_ext - mod * paso
        PHI, RAD = np.meshgrid(phi, rad)
        modo_grilla = np.meshgrid(sit, mod)
        # Precisión simple (float32) solo para la gráfica: basta para dibujar y el archivo pesa la mitad.
        f32 = lambda a: np.asarray(a, dtype=np.float32)
        Z, S = f32(Z), f32(S)
        trazas.append(go.Surface(
            x=f32(RAD * np.cos(PHI)), y=f32(RAD * np.sin(PHI)), z=Z, surfacecolor=S,
            cmin=-1.0, cmax=1.0, name=f"Región {nombre}",
            colorscale=[[0.0, "#c0392b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]],
            showscale=(r == 0),
            colorbar=dict(title="cos 2θ", thickness=15, len=0.6, tickvals=[-1, 0, 1],
                          ticktext=["todo b", "mitad y mitad", "todo a"]),
            customdata=f32(np.dstack([modo_grilla[0], modo_grilla[1], S])),
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1, roughness=0.7),
            lightposition=dict(x=100, y=200, z=1000),
            hovertemplate=f"Región {nombre}<br>" + "sitio %{customdata[0]:.1f}"
                          "<br>modo %{customdata[1]:.1f}<br>u²+v² = %{z:.4g}"
                          "<br>cos 2θ = %{customdata[2]:.3f}<extra></extra>"))

    # Cortes entre regiones (líneas radiales en el piso) y nombres de las regiones
    R_min = R_ext - (kmax - 1) * paso
    for _, _, _, sitios in regiones:
        ang = 2 * np.pi * (min(sitios) - 1) / N
        trazas.append(go.Scatter3d(x=[R_min * np.cos(ang), 1.08 * np.cos(ang)],
                                   y=[R_min * np.sin(ang), 1.08 * np.sin(ang)], z=[0, 0],
                                   mode="lines", line=dict(color="#383835", width=4),
                                   hoverinfo="skip", showlegend=False))
    for nombre, _, _, sitios in regiones:
        medio = 2 * np.pi * ((min(sitios) + max(sitios)) / 2 - 0.5) / N
        trazas.append(go.Scatter3d(x=[1.2 * np.cos(medio)], y=[1.2 * np.sin(medio)], z=[0],
                                   mode="text", text=[nombre], textfont=dict(size=18),
                                   hoverinfo="skip", showlegend=False))
    return trazas, zmax


def profundidades(H, niveles=(0.9, 0.99, 0.999), peso_min=0.1):
    """
    Contornos de los pétalos de altura de una región.
    Para cada modo m (fila de H), p_m(x) = H[m][x] / 2 es una distribución sobre los sitios de
    la región. La región se parte por la mitad: el lado izquierdo pertenece al primer corte y
    el derecho al segundo. En cada lado, la profundidad del nivel l es la distancia al corte
    (en sitios, con interpolación lineal dentro de cada sitio) que encierra la fracción l del
    peso que el modo tiene en ese lado. Si un lado tiene menos de peso_min del peso del modo,
    el modo no tiene pétalo en ese corte (None).
    Devuelve P[lado][nivel][m], con lado 0 = primer corte y 1 = segundo corte.
    """
    import numpy as np
    H = np.asarray(H, dtype=float)
    n = H.shape[1]
    mitad = n // 2
    P = [[[None] * len(H) for _ in niveles] for _ in range(2)]
    for m, fila in enumerate(H):
        p = fila / fila.sum()
        lados = (p[:mitad], p[mitad:][::-1])        # ordenados desde su corte hacia adentro
        for s, q in enumerate(lados):
            w = q.sum()
            if w < peso_min:
                continue
            G = np.concatenate([[0.0], np.cumsum(q / w)])   # G[j] = peso hasta la distancia j
            for i, l in enumerate(niveles):
                j = min(int(np.searchsorted(G, l)), len(q))    # G[j-1] < l <= G[j]
                dentro = (l - G[j - 1]) / (G[j] - G[j - 1]) if G[j] > G[j - 1] else 1.0
                P[s][i][m] = (j - 1) + dentro
    return P


COLORES_PETALOS = {          # por región: del nivel más bajo (fuerte) al más alto (claro)
    0: ["#1f5fae", "#6fa3e0", "#c5daf3", "#e6eef9"],
    1: ["#c4511f", "#f0965f", "#f9d3b8", "#fcebdf"],
    2: ["#237a37", "#6cc07f", "#c8ead0", "#e5f5e8"],
}
GRISES_PETALOS = ["#3d3d3a", "#8a8984", "#c9c8c2", "#e4e3de"]


def _suavizar(valores):
    """Promedio (1/4, 1/2, 1/4) de cada valor con sus vecinos, dentro de cada tramo sin None;
    en los extremos de un tramo, (2/3, 1/3) con el único vecino."""
    suave = list(valores)
    for i, v in enumerate(valores):
        if v is None:
            continue
        izq = valores[i - 1] if i > 0 else None
        der = valores[i + 1] if i + 1 < len(valores) else None
        if izq is not None and der is not None:
            suave[i] = (izq + 2 * v + der) / 4
        elif izq is not None or der is not None:
            suave[i] = (2 * v + (izq if izq is not None else der)) / 3
    return suave


def _sector(a0, a1, r0, r1, puntos=8):
    """Polígono (xs, ys) de un sector de anillo entre los ángulos a0, a1 y los radios r0, r1."""
    import numpy as np
    ang = np.linspace(a0, a1, max(2, puntos))
    xs = list(r1 * np.cos(ang)) + list(r0 * np.cos(ang[::-1]))
    ys = list(r1 * np.sin(ang)) + list(r0 * np.sin(ang[::-1]))
    return xs + [xs[0], None], ys + [ys[0], None]


def figura_petalos(regiones, N, niveles=(0.9, 0.99, 0.999), titulo="", max_modos=None,
                   peso_min=0.1, lineas=True):
    """
    Pétalos de altura en 2D, vistos desde arriba (misma geometría que el cilindro: ángulo =
    sitio, radio = modo, el modo 0 afuera). Cada anillo es un sector de corona; desde cada corte
    hacia adentro se rellenan bandas tipo mapa topográfico: de 0 al primer nivel (color fuerte),
    del primero al segundo (medio), etc. (ver profundidades). Lo que queda en gris muy claro está
    más allá del último nivel. Sin interpolar entre modos.
    lineas: además, una curva suave en cada frontera entre bandas. Es solo una guía visual: la
    profundidad de cada anillo se promedia con la de sus vecinos (pesos 1/4, 1/2, 1/4) y la
    curva se traza con una spline; las bandas de color siguen siendo los datos de cada anillo.
    regiones: lista de (nombre, H, C, sitios), como en trazas_cilindro.
    """
    import numpy as np
    k_reg = [len(H) if max_modos is None else min(len(H), max_modos) for _, H, _, _ in regiones]
    kmax = max(k_reg)
    R_ext, R_int = 1.0, 0.35                          # igual que en trazas_cilindro
    paso = (R_ext - R_int) / max(kmax - 1, 1)
    medio = paso / 2
    niveles = tuple(sorted(niveles))
    etiquetas = [f"0–{100 * niveles[0]:g} %"] + [f"{100 * a:g}–{100 * b:g} %"
                                                  for a, b in zip(niveles, niveles[1:])]
    fig = go.Figure()

    for r, ((nombre, H, _, sitios), k) in enumerate(zip(regiones, k_reg)):
        colores = COLORES_PETALOS[r % 3]
        n = len(sitios)
        a_ini = 2 * np.pi * (min(sitios) - 1) / N     # primer corte
        a_fin = 2 * np.pi * max(sitios) / N           # segundo corte
        # Fondo de la región: más allá del último nivel
        xs, ys = [], []
        for m in range(k):
            rad = R_ext - m * paso
            x, y = _sector(a_ini, a_fin, rad - medio, rad + medio, puntos=max(8, n))
            xs += x; ys += y
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=colores[3],
                                 line=dict(width=0.6, color=colores[3]), mode="lines",
                                 hoverinfo="skip", showlegend=False))
        # Bandas
        P = profundidades(H[:k], niveles, peso_min)
        bandas = [([], []) for _ in niveles]
        marcas_x, marcas_y, marcas_t = [], [], []
        for s, (corte, signo) in enumerate(((a_ini, 1), (a_fin, -1))):
            for m in range(k):
                if P[s][0][m] is None:
                    continue
                rad = R_ext - m * paso
                d_prev = 0.0
                for b in range(len(niveles)):
                    d = P[s][b][m]
                    a0 = corte + signo * 2 * np.pi * d_prev / N
                    a1 = corte + signo * 2 * np.pi * d / N
                    x, y = _sector(a0, a1, rad - medio, rad + medio)
                    bandas[b][0].extend(x); bandas[b][1].extend(y)
                    ang = (a0 + a1) / 2
                    marcas_x.append(rad * np.cos(ang)); marcas_y.append(rad * np.sin(ang))
                    marcas_t.append(f"Región {nombre}, modo {m}, "
                                    f"{'primer' if s == 0 else 'segundo'} corte<br>"
                                    f"{etiquetas[b]} del peso: de {d_prev:.2f} a {d:.2f} sitios")
                    d_prev = d
        for b, (xs, ys) in enumerate(bandas):
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=colores[b],
                                     line=dict(width=0.6, color=colores[b]), mode="lines",
                                     hoverinfo="skip", showlegend=False))
        # Curvas de contorno suaves en cada frontera entre bandas
        if lineas:
            for s, (corte, signo) in enumerate(((a_ini, 1), (a_fin, -1))):
                for b in range(len(niveles)):
                    xs, ys = [], []
                    suave = _suavizar(P[s][b])
                    for m in range(k):
                        d = suave[m]
                        if d is None:
                            xs.append(None); ys.append(None)
                            continue
                        rad = R_ext - m * paso
                        ang = corte + signo * 2 * np.pi * d / N
                        xs.append(rad * np.cos(ang)); ys.append(rad * np.sin(ang))
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                             showlegend=False,
                                             line=dict(color="#2b2b28", width=1.3,
                                                       shape="spline", smoothing=0.8)))
        # Marcas invisibles, solo para ver los números al pasar el mouse
        fig.add_trace(go.Scatter(x=marcas_x, y=marcas_y, mode="markers", text=marcas_t,
                                 hoverinfo="text", marker=dict(size=6, opacity=0),
                                 showlegend=False))
        # Cortes y nombre de la región
        for a in (a_ini, a_fin):
            fig.add_trace(go.Scatter(x=[(R_int - medio) * np.cos(a), 1.08 * np.cos(a)],
                                     y=[(R_int - medio) * np.sin(a), 1.08 * np.sin(a)],
                                     mode="lines", line=dict(color="#383835", width=2),
                                     hoverinfo="skip", showlegend=False))
        a_med = 2 * np.pi * ((min(sitios) + max(sitios)) / 2 - 0.5) / N
        fig.add_trace(go.Scatter(x=[1.18 * np.cos(a_med)], y=[1.18 * np.sin(a_med)],
                                 mode="text", text=[nombre], textfont=dict(size=18),
                                 hoverinfo="skip", showlegend=False))

    # Leyenda (en grises: el mismo significado para las tres regiones)
    for b, etiqueta in enumerate(etiquetas + [f"más de {100 * niveles[-1]:g} %"]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=etiqueta,
                                 marker=dict(symbol="square", size=14,
                                             color=GRISES_PETALOS[min(b, 3)])))
    lim = 1.3
    fig.update_layout(
        title=dict(text=titulo, x=0.5), plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-lim, lim]),
        yaxis=dict(visible=False, range=[-lim, lim], scaleanchor="x"),
        legend=dict(title="peso desde el corte", x=1.0, y=0.5, xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10), width=950, height=800)
    return fig


def profundidades_radiales(H, niveles=(0.9, 0.99, 0.999)):
    """
    Pétalos a ángulo constante. Por ortogonalidad de O_R, cada columna de H suma 2:
    q_x(m) = H[m][x] / 2 es una distribución sobre los modos para cada sitio x (la fracción del
    sitio x que lleva el modo m). Acumulando desde el modo 0 (el más entrelazado, anillo
    exterior) hacia adentro, el nivel l se alcanza en el "anillo" t_l(x): el anillo m ocupa el
    intervalo [m, m + 1) y se interpola linealmente dentro de él.
    H debe tener todos los modos de la región (si no, la suma no llega a 1).
    Devuelve T[nivel][x] (float entre 0 y el número de modos).
    """
    import numpy as np
    H = np.asarray(H, dtype=float)
    q = H / H.sum(axis=0, keepdims=True)
    Q = np.vstack([np.zeros(H.shape[1]), np.cumsum(q, axis=0)])   # Q[m] = peso antes del anillo m
    T = []
    for l in niveles:
        fila = []
        for x in range(H.shape[1]):
            m = min(int(np.searchsorted(Q[:, x], l)), H.shape[0])     # Q[m-1] < l <= Q[m]
            dentro = (l - Q[m - 1, x]) / (Q[m, x] - Q[m - 1, x]) if Q[m, x] > Q[m - 1, x] else 1.0
            fila.append((m - 1) + dentro)
        T.append(fila)
    return T


def figura_petalos_sitio(regiones, N, niveles=(0.9, 0.99, 0.999), titulo="", max_modos=None,
                         lineas=True):
    """
    Pétalos a ángulo constante, en 2D vistos desde arriba (misma geometría que el cilindro).
    Para cada sitio (sector angular), desde el anillo exterior hacia adentro se rellenan bandas
    tipo mapa topográfico: hasta el anillo donde se acumula el primer nivel del peso del sitio
    (color fuerte), del primero al segundo (medio), etc. (ver profundidades_radiales).
    Si un nivel cae más allá del último anillo dibujado (max_modos), en el borde interior de ese
    sitio se dibuja una franja gris: oscura si ya el primer nivel queda fuera, más clara si solo
    los siguientes.
    lineas: curvas suaves en las fronteras entre bandas (guía visual: la posición de cada sitio
    se promedia con la de sus vecinos y se traza con una spline).
    regiones: lista de (nombre, H, C, sitios) con TODOS los modos de cada región.
    """
    import numpy as np
    k_reg = [len(H) if max_modos is None else min(len(H), max_modos) for _, H, _, _ in regiones]
    kmax = max(k_reg)
    R_ext, R_int = 1.0, 0.35                          # igual que en trazas_cilindro
    paso = (R_ext - R_int) / max(kmax - 1, 1)
    borde_ext = R_ext + paso / 2                      # radio del borde exterior del anillo 0
    radio = lambda t: borde_ext - t * paso            # posición t (en anillos) -> radio
    niveles = tuple(sorted(niveles))
    etiquetas = [f"0–{100 * niveles[0]:g} %"] + [f"{100 * a:g}–{100 * b:g} %"
                                                  for a, b in zip(niveles, niveles[1:])]
    fig = go.Figure()

    for r, ((nombre, H, _, sitios), k) in enumerate(zip(regiones, k_reg)):
        colores = COLORES_PETALOS[r % 3]
        sitios = list(sitios)
        T = profundidades_radiales(H, niveles)
        borde_int = radio(k)
        bordes = [2 * np.pi * (x - 1) / N for x in sitios] + [2 * np.pi * sitios[-1] / N]
        # Fondo: más allá del último nivel
        x, y = _sector(bordes[0], bordes[-1], borde_int, borde_ext, puntos=max(8, len(sitios)))
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=colores[3],
                                 line=dict(width=0.6, color=colores[3]), mode="lines",
                                 hoverinfo="skip", showlegend=False))
        bandas = [([], []) for _ in niveles]
        fuera = [([], []) for _ in niveles]
        marcas_x, marcas_y, marcas_t = [], [], []
        for i in range(len(sitios)):
            a0, a1 = bordes[i], bordes[i + 1]
            t_prev = 0.0
            for b in range(len(niveles)):
                t = min(T[b][i], k)
                if t > t_prev:
                    xs, ys = _sector(a0, a1, radio(t), radio(t_prev), puntos=3)
                    bandas[b][0].extend(xs); bandas[b][1].extend(ys)
                t_prev = t
            primero_fuera = next((b for b in range(len(niveles)) if T[b][i] > k), None)
            if primero_fuera is not None:
                xs, ys = _sector(a0, a1, borde_int - 0.045, borde_int - 0.01, puntos=3)
                fuera[primero_fuera][0].extend(xs); fuera[primero_fuera][1].extend(ys)
            ang = (a0 + a1) / 2
            marcas_x.append(radio(min(T[0][i], k) / 2) * np.cos(ang))
            marcas_y.append(radio(min(T[0][i], k) / 2) * np.sin(ang))
            marcas_t.append(f"Región {nombre}, sitio {sitios[i]}<br>" + "<br>".join(
                f"{100 * l:g} % del sitio en los anillos 0–{T[b][i]:.2f}"
                + (f"  (más allá del anillo {k - 1} dibujado)" if T[b][i] > k else "")
                for b, l in enumerate(niveles)))
        for b, (xs, ys) in enumerate(bandas):
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=colores[b],
                                     line=dict(width=0.6, color=colores[b]), mode="lines",
                                     hoverinfo="skip", showlegend=False))
        for b, (xs, ys) in enumerate(fuera):
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                                         fillcolor=GRISES_PETALOS[min(b, 3)],
                                         line=dict(width=0.6, color=GRISES_PETALOS[min(b, 3)]),
                                         mode="lines", hoverinfo="skip", showlegend=False))
        if lineas:
            angs = [(bordes[i] + bordes[i + 1]) / 2 for i in range(len(sitios))]
            for b in range(len(niveles)):
                dentro = [T[b][i] if T[b][i] <= k else None for i in range(len(sitios))]
                suave = _suavizar(dentro)
                xs = [None if t is None else radio(t) * np.cos(a) for t, a in zip(suave, angs)]
                ys = [None if t is None else radio(t) * np.sin(a) for t, a in zip(suave, angs)]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                         showlegend=False,
                                         line=dict(color="#2b2b28", width=1.3, shape="spline",
                                                   smoothing=0.8)))
        fig.add_trace(go.Scatter(x=marcas_x, y=marcas_y, mode="markers", text=marcas_t,
                                 hoverinfo="text", marker=dict(size=6, opacity=0),
                                 showlegend=False))
        for a in (bordes[0], bordes[-1]):
            fig.add_trace(go.Scatter(x=[(borde_int - 0.05) * np.cos(a), 1.08 * np.cos(a)],
                                     y=[(borde_int - 0.05) * np.sin(a), 1.08 * np.sin(a)],
                                     mode="lines", line=dict(color="#383835", width=2),
                                     hoverinfo="skip", showlegend=False))
        a_med = 2 * np.pi * ((sitios[0] + sitios[-1]) / 2 - 0.5) / N
        fig.add_trace(go.Scatter(x=[1.18 * np.cos(a_med)], y=[1.18 * np.sin(a_med)],
                                 mode="text", text=[nombre], textfont=dict(size=18),
                                 hoverinfo="skip", showlegend=False))

    for b, etiqueta in enumerate(etiquetas + [f"más de {100 * niveles[-1]:g} %"]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=etiqueta,
                                 marker=dict(symbol="square", size=14,
                                             color=GRISES_PETALOS[min(b, 3)])))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             name="franja interior: nivel más allá<br>del último anillo",
                             marker=dict(symbol="line-ew", size=14,
                                         line=dict(width=4, color=GRISES_PETALOS[0]))))
    lim = 1.3
    fig.update_layout(
        title=dict(text=titulo, x=0.5), plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-lim, lim]),
        yaxis=dict(visible=False, range=[-lim, lim], scaleanchor="x"),
        legend=dict(title="peso del sitio, desde<br>el anillo exterior", x=1.0, y=0.5,
                    xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10), width=950, height=800)
    return fig


def graficar_petalos_sitio(regiones, N, niveles=(0.9, 0.99, 0.999), titulo="", archivo=None,
                           max_modos=None, lineas=True):
    """Pétalos a ángulo constante (ver figura_petalos_sitio): los guarda en 'archivo' (.html)
    o, si no se da, los muestra."""
    fig = figura_petalos_sitio(regiones, N, niveles, titulo, max_modos, lineas)
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def graficar_petalos(regiones, N, niveles=(0.9, 0.99, 0.999), titulo="", archivo=None,
                     max_modos=None, peso_min=0.1, lineas=True):
    """Pétalos de altura en 2D (ver figura_petalos): los guarda en 'archivo' (.html) o, si no
    se da, los muestra."""
    fig = figura_petalos(regiones, N, niveles, titulo, max_modos, peso_min, lineas)
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def _layout_cilindro(titulo, zmax):
    sin_eje = dict(showticklabels=False, title="", showgrid=False, zeroline=False,
                   showbackground=False)
    return dict(
        title=dict(text=titulo, x=0.5),
        scene=dict(xaxis=sin_eje, yaxis=sin_eje,
                   zaxis=dict(title="u² + v²", range=[0, zmax], showbackground=False),
                   aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=0.45),
                   camera=dict(eye=dict(x=0.0, y=-1.6, z=1.1))),
        margin=dict(l=0, r=0, t=50, b=0), width=1100, height=850)


def figura_cilindro(regiones, N, titulo="", max_modos=None, refinar=1):
    """Figura (plotly) de la gráfica cilíndrica de un caso (ver trazas_cilindro)."""
    trazas, zmax = trazas_cilindro(regiones, N, max_modos, refinar)
    fig = go.Figure(trazas)
    fig.update_layout(**_layout_cilindro(titulo, zmax))
    return fig


def graficar_cilindro(regiones, N, titulo="", archivo=None, max_modos=None, refinar=1):
    """Gráfica cilíndrica de un caso (ver trazas_cilindro): la guarda en 'archivo' (.html)
    o, si no se da, la muestra."""
    fig = figura_cilindro(regiones, N, titulo, max_modos, refinar)
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def graficar_cilindro_deslizador(casos, N, nombre_parametro="", titulo="", archivo=None,
                                 max_modos=None, refinar=1):
    """
    Una sola gráfica cilíndrica con un deslizador que recorre varios casos.
    casos: lista de (etiqueta, regiones), con regiones como en trazas_cilindro.
    La altura máxima del eje z es la misma para todos los casos, para poder compararlos.
    """
    cuadros = []
    zmax = 0.0
    for etiqueta, regiones in casos:
        trazas, z = trazas_cilindro(regiones, N, max_modos, refinar)
        cuadros.append(go.Frame(data=trazas, name=etiqueta))
        zmax = max(zmax, z)

    fig = go.Figure(data=cuadros[0].data, frames=cuadros)
    fig.update_layout(**_layout_cilindro(titulo, zmax))
    quieto = {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
              "transition": {"duration": 0}}
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=70),
        sliders=[dict(
            active=0, x=0.05, len=0.9, y=0.02, pad=dict(t=10),
            currentvalue=dict(prefix="", font=dict(size=16)),
            steps=[dict(method="animate", label=c.name, args=[[c.name], quieto])
                   for c in cuadros])],
        updatemenus=[dict(
            type="buttons", direction="left", x=0.05, y=0.1, xanchor="left", showactive=False,
            buttons=[
                dict(label="▶ reproducir", method="animate",
                     args=[None, {"mode": "immediate", "fromcurrent": True,
                                  "frame": {"duration": 800, "redraw": True},
                                  "transition": {"duration": 0}}]),
                dict(label="❚❚ pausa", method="animate", args=[[None], quieto])])])
    if archivo:
        fig.write_html(archivo)
    else:
        fig.show()


def leer_matriz(archivo):
    with open(archivo) as f:
        filas = [[mp.mpf(x) for x in linea.split()] for linea in f if linea.strip()]
    return mp.matrix(filas)


def leer_lista(archivo):
    with open(archivo) as f:
        return [mp.mpf(x) for x in f if x.strip()]


def graficar_caso(gamma, h, N, NA, NB, dps, max_modos=None, refinar=1):
    """Lee O_A, O_B, O_C de Resultados_diag_local para (gamma, h, N, NA, NB) y guarda en
    Resultados_modos una superficie 3D de la función de modos de cada región, y una gráfica
    con las tres regiones juntas. dps: precisión con la que se calculó el caso; se agrega al
    nombre de las gráficas (..._dps70_ABC_cilindro.html). Los archivos de entrada se buscan
    primero en Resultados_diag_local/dps<dps> y, si no están, en Resultados_diag_local."""
    mp.dps = dps
    NC = N - NA - NB
    programas = os.path.dirname(os.path.abspath(__file__))
    entrada = os.path.join(programas, "Resultados_diag_local", f"dps{dps}")
    if not os.path.isdir(entrada):
        entrada = os.path.join(programas, "Resultados_diag_local")
    salida = os.path.join(programas, "Resultados_modos")
    os.makedirs(salida, exist_ok=True)

    prefijo_entrada = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}_A{NA}_B{NB}_C{NC}"
    prefijo = f"{prefijo_entrada}_dps{dps}"
    particion = {"A": range(0, NA), "B": range(NA, NA + NB), "C": range(NA + NB, N)}
    juntas = []
    for nombre, R in particion.items():
        archivo_O = os.path.join(entrada, f"{prefijo_entrada}_O_{nombre}.txt")
        if not os.path.exists(archivo_O):
            base = os.path.join(programas, "Resultados_diag_local")
            otros = [d for d in sorted(os.listdir(base))
                     if os.path.exists(os.path.join(base, d, f"{prefijo_entrada}_O_{nombre}.txt"))]
            pista = f" Sí está en: {', '.join(otros)}." if otros else ""
            raise FileNotFoundError(f"No encontré {prefijo_entrada}_O_{nombre}.txt en {entrada}.{pista}")
        destino = os.path.join(salida, f"{prefijo}_region{nombre}_3D.html")
        H, C = funcion_de_modos(leer_matriz(archivo_O))
        graficar_funcion_de_modos(H, C, sitios=[i + 1 for i in R], max_modos=max_modos, archivo=destino, refinar=refinar,
                                  titulo=f"Región {nombre}   (γ={gamma}, h={h}, N={N}, "
                                         f"A|B|C = {NA}|{NB}|{NC}, dps={dps})")
        print(f"Región {nombre}: {destino}", flush=True)
        juntas.append((nombre, H, C, [i + 1 for i in R]))

    # Las tres regiones en una misma gráfica
    destino = os.path.join(salida, f"{prefijo}_ABC_3D.html")
    graficar_regiones_juntas(juntas, max_modos=max_modos, refinar=refinar, archivo=destino,
                             titulo=f"Regiones A | B | C   (γ={gamma}, h={h}, N={N}, "
                                    f"A|B|C = {NA}|{NB}|{NC}, dps={dps})")
    print(f"Las tres juntas: {destino}", flush=True)

    # Las tres regiones sobre el anillo (coordenadas cilíndricas)
    destino = os.path.join(salida, f"{prefijo}_ABC_cilindro.html")
    graficar_cilindro(juntas, N, max_modos=max_modos, refinar=refinar, archivo=destino,
                      titulo=f"Anillo, regiones A | B | C   (γ={gamma}, h={h}, N={N}, "
                             f"A|B|C = {NA}|{NB}|{NC}, dps={dps})")
    print(f"Anillo (cilíndrico): {destino}", flush=True)


def graficar_caso_biparticion(gamma, h, N, NS, inicio=0, dps=70, max_modos=None, refinar=1,
                              etiquetas=("S", "T")):
    """Lo mismo que graficar_caso, para una bipartición S | T del anillo: S = arco de NS sitios
    que empieza en el sitio 'inicio' (numerados 0..N-1), T = el resto (ver diag_local).
    Lee O_S y O_T de Resultados_diag_local/dps<dps> y guarda en Resultados_modos:
        <prefijo>_dps<dps>_regionS_3D.html, ..._regionT_3D.html, ..._ST_3D.html, ..._ST_cilindro.html
    con <prefijo> = anillo_par_N{N}_gamma{gamma}_h{h}_bip_ini{inicio}_NS{NS}.
    etiquetas: solo cambian los nombres de S y T en las gráficas (p. ej. ("A", "BC")).
    Los sitios se numeran desde 1 sin dar la vuelta: si T cruza el enlace (N, 1), sus sitios
    siguen después de N (N+1, N+2, ...), para que cada arco se dibuje seguido."""
    mp.dps = dps
    inicio %= N
    NT = N - NS
    programas = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(programas, "Resultados_diag_local")
    entrada = os.path.join(base, f"dps{dps}")
    salida = os.path.join(programas, "Resultados_modos")
    os.makedirs(salida, exist_ok=True)

    prefijo_entrada = f"anillo_par_N{N}_gamma{float(gamma)}_h{float(h)}_bip_ini{inicio}_NS{NS}"
    prefijo = f"{prefijo_entrada}_dps{dps}"
    nombre = f"{etiquetas[0]} | {etiquetas[1]}"
    datos = (f"γ={gamma}, h={h}, N={N}, {nombre} = {NS}|{NT}, "
             f"{etiquetas[0]} desde el sitio {inicio + 1}, dps={dps}")
    sitios = {"S": list(range(inicio + 1, inicio + NS + 1)),
              "T": list(range(inicio + NS + 1, inicio + N + 1))}
    juntas = []
    for r, etiqueta in zip("ST", etiquetas):
        archivo_O = os.path.join(entrada, f"{prefijo_entrada}_O_{r}.txt")
        if not os.path.exists(archivo_O):
            otros = [d for d in sorted(os.listdir(base))
                     if os.path.exists(os.path.join(base, d, f"{prefijo_entrada}_O_{r}.txt"))]
            pista = f" Sí está en: {', '.join(otros)}." if otros else ""
            raise FileNotFoundError(f"No encontré {prefijo_entrada}_O_{r}.txt en {entrada}.{pista}")
        destino = os.path.join(salida, f"{prefijo}_region{r}_3D.html")
        H, C = funcion_de_modos(leer_matriz(archivo_O))
        graficar_funcion_de_modos(H, C, sitios=sitios[r], max_modos=max_modos, archivo=destino,
                                  refinar=refinar, titulo=f"Región {etiqueta}   ({datos})")
        print(f"Región {etiqueta}: {destino}", flush=True)
        juntas.append((etiqueta, H, C, sitios[r]))

    # Las dos regiones en una misma gráfica
    destino = os.path.join(salida, f"{prefijo}_ST_3D.html")
    graficar_regiones_juntas(juntas, max_modos=max_modos, refinar=refinar, archivo=destino,
                             titulo=f"Regiones {nombre}   ({datos})")
    print(f"Las dos juntas: {destino}", flush=True)

    # Las dos regiones sobre el anillo (coordenadas cilíndricas)
    destino = os.path.join(salida, f"{prefijo}_ST_cilindro.html")
    graficar_cilindro(juntas, N, max_modos=max_modos, refinar=refinar, archivo=destino,
                      titulo=f"Anillo, bipartición {nombre}   ({datos})")
    print(f"Anillo (cilíndrico): {destino}", flush=True)


if __name__ == "__main__":
    gamma = 0.5
    h = 0.3
    N = 200
    NA, NB = 20, 20          # C es el resto: N - NA - NB
    dps = 70                 # precisión con la que se calculó el caso

    graficar_caso(gamma, h, N, NA, NB, dps)