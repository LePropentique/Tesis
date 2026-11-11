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