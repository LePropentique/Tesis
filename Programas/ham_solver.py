"""
Cadena XY con campo transverso, anillo de espines, sector de paridad par.
Devuelve la ortogonal O_tilde (Majoranas de los sitios -> Majoranas de los modos)
en precisión arbitraria.

    H = sum_i [ (1+gamma) S^x_i S^x_{i+1} + (1-gamma) S^y_i S^y_{i+1} ] - h sum_i S^z_i

Sector par: fermiones antiperiódicos, c_{N+1} = -c_1.
Método de Lieb:  A - B = phi^T diag(Lambda) psi   (SVD),   O_tilde = [[phi, 0], [0, psi]]
Majoranas: a_i = c_i + c_i^†,  b_i = i (c_i - c_i^†),  orden (a_1..a_N | b_1..b_N).

Requiere: numpy, mpmath   (pip install numpy mpmath)
"""
import os
import time
import numpy as np
from mpmath import mp


def build_AB_par(N, gamma, h):
    """A (simétrica) y B (antisimétrica) del sector par del anillo."""
    A = -h * np.eye(N) + 0.5 * (np.eye(N, k=1) + np.eye(N, k=-1))
    B = 0.5 * gamma * (np.eye(N, k=1) - np.eye(N, k=-1))
    A[0, N - 1] = A[N - 1, 0] = -0.5
    B[N - 1, 0] = -0.5 * gamma
    B[0, N - 1] = 0.5 * gamma
    return A, B


def solve_phi_psi(A, B):
    """(A-7) de Lieb como SVD:  A - B = phi^T diag(Lam) psi."""
    M = mp.matrix(A.tolist()) - mp.matrix(B.tolist())
    U, Lam, V = mp.svd_r(M)          # M = U diag(Lam) V
    return Lam, U.T, V


def O_tilde(phi, psi):
    """O_tilde = [[phi, 0], [0, psi]]."""
    N = phi.rows
    Ot = mp.zeros(2 * N, 2 * N)
    for i in range(N):
        for j in range(N):
            Ot[i, j] = phi[i, j]
            Ot[N + i, N + j] = psi[i, j]
    return Ot


def ortogonal_anillo_par(gamma, h, dps, N=200):
    """Recibe gamma, h y la precisión (dígitos); devuelve O_tilde, Lambda y E0."""
    mp.dps = dps
    A, B = build_AB_par(N, gamma, h)
    Lam, phi, psi = solve_phi_psi(A, B)
    Ot = O_tilde(phi, psi)
    E0 = -mp.fsum(Lam) / 2           # incluye la constante N h / 2
    return Ot, Lam, E0


def guardar(nombre, Ot, Lam, dps):
    """Guarda O_tilde y Lambda como texto con todos los dígitos."""
    with open(nombre + "_O_tilde.txt", "w") as f:
        for i in range(Ot.rows):
            f.write(" ".join(mp.nstr(Ot[i, j], dps) for j in range(Ot.cols)) + "\n")
    with open(nombre + "_Lambda.txt", "w") as f:
        for k in range(len(Lam)):
            f.write(mp.nstr(Lam[k], dps) + "\n")


if __name__ == "__main__":
    N = 200
    DPS = 70
    DPS_CRITICO = 100                # precisión en los puntos críticos (h = 1)
    HORAS = 6.3                      # presupuesto de tiempo

    # Barrido en h, ordenado de grueso a fino:
    # si el tiempo no alcanza para todos, igual queda cubierto todo el rango.
    hs = [1.0, 0.5, 1.5, 0.0,
          0.25, 0.75, 1.25,
          0.9, 1.1, 0.95, 1.05,
          0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1.2, 1.3, 1.4]
    casos = ([(1.0, h) for h in hs] + [(0.5, h) for h in hs]
             + [(0.75, h) for h in hs] + [(0.25, h) for h in hs])

    t_inicio = time.time()
    duraciones = []
    for gamma, h in casos:
        dps = DPS_CRITICO if h == 1.0 else DPS
        carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "Resultados_Ham_Solver", f"dps{dps}")
        os.makedirs(carpeta, exist_ok=True)
        nombre = os.path.join(carpeta, f"anillo_par_N{N}_gamma{gamma}_h{h}")
        if os.path.exists(nombre + "_O_tilde.txt"):
            print(f"gamma={gamma}, h={h} (dps={dps}): ya existe, lo salto", flush=True)
            continue

        transcurrido = time.time() - t_inicio
        estimado = sum(duraciones) / len(duraciones) if duraciones else 0.0
        if transcurrido + estimado > HORAS * 3600:
            print(f"Se acabó el tiempo ({transcurrido / 3600:.2f} h). Quedan casos sin correr.", flush=True)
            break

        t0 = time.time()
        Ot, Lam, E0 = ortogonal_anillo_par(gamma, h, dps, N)
        guardar(nombre, Ot, Lam, dps)
        duraciones.append(time.time() - t0)
        print(f"gamma={gamma}, h={h} (dps={dps}):  E0/N = {mp.nstr(E0 / N, 15)},  "
              f"Lambda_min = {mp.nstr(Lam[N - 1], 6)},  ({duraciones[-1]:.0f} s, "
              f"total {(time.time() - t_inicio) / 3600:.2f} h)", flush=True)

    print(f"Terminado: {len(duraciones)} casos nuevos en {(time.time() - t_inicio) / 3600:.2f} h")