"""
Detecta con cuántos dígitos (dps) se guardó cada resultado y lo mueve a una subcarpeta
dps<n> dentro de su carpeta de resultados:

    Resultados_Ham_Solver/dps50/...,   Resultados_Ham_Solver/dps100/...
    Resultados_covariance_matrix/dps50/..., etc.
    Resultados_diag_local/dps50/..., etc.

El número de dígitos se lee del propio archivo: es la mayor cantidad de cifras
significativas entre los números guardados (los archivos se escribieron con mp.nstr(x, dps)).
Todos los archivos de un mismo caso se mueven juntos.

Uso: poner este archivo en la carpeta Programas y ejecutarlo. Primero muestra la lista
de lo que encontró y pregunta antes de mover nada.
"""
import os
import shutil

# carpeta de resultados -> sufijo del archivo con el que se detecta el dps del caso
CARPETAS = {
    "Resultados_Ham_Solver": "_Lambda.txt",
    "Resultados_covariance_matrix": "_M.txt",
    "Resultados_diag_local": "_lambda_A.txt",
}


def cifras(numero):
    """Cifras significativas de un número escrito como texto (p. ej. '-1.2345e-15' -> 5)."""
    mantisa = numero.strip().lstrip("+-").lower().split("e")[0].replace(".", "")
    return len(mantisa.lstrip("0")) or 1


def dps_de_archivo(ruta, max_lineas=3):
    """Mayor número de cifras entre los números de las primeras líneas del archivo."""
    mayor = 0
    with open(ruta) as f:
        for k, linea in enumerate(f):
            if k >= max_lineas:
                break
            for x in linea.split():
                mayor = max(mayor, cifras(x))
    return mayor


def casos_en(carpeta, sufijo):
    """Lista de (prefijo del caso, dps) para los archivos sueltos de la carpeta."""
    casos = []
    for nombre in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, nombre)
        if os.path.isfile(ruta) and nombre.endswith(sufijo):
            casos.append((nombre[: -len(sufijo)], dps_de_archivo(ruta)))
    return casos


def mover_caso(carpeta, prefijo, dps):
    destino = os.path.join(carpeta, f"dps{dps}")
    os.makedirs(destino, exist_ok=True)
    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if os.path.isfile(ruta) and nombre.startswith(prefijo + "_"):
            shutil.move(ruta, os.path.join(destino, nombre))


if __name__ == "__main__":
    programas = os.path.dirname(os.path.abspath(__file__))
    plan = []
    for carpeta_nombre, sufijo in CARPETAS.items():
        carpeta = os.path.join(programas, carpeta_nombre)
        if not os.path.isdir(carpeta):
            continue
        casos = casos_en(carpeta, sufijo)
        if not casos:
            continue
        print(f"\n{carpeta_nombre}:")
        for prefijo, dps in casos:
            print(f"   dps{dps:<4} {prefijo}")
            plan.append((carpeta, prefijo, dps))

    if not plan:
        print("No encontré archivos sueltos para organizar.")
    elif input("\n¿Mover estos casos a sus carpetas dps<n>? (s/n): ").strip().lower() == "s":
        for carpeta, prefijo, dps in plan:
            mover_caso(carpeta, prefijo, dps)
        print("Listo.")
    else:
        print("No moví nada.")