# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO: ANÁLISIS CINEMÁTICO DEL CENTROIDE
# ══════════════════════════════════════════════════════════════════════════════
# Referencia de escala:
#   Puntos medios del ROI: [326, 130] (medio derecho) y [281, 130] (medio izquierdo)
#   Distancia horizontal en píxeles: |326 - 281| = 45 px
#   Distancia real conocida: 4 metros
#   => Factor de escala: 4.0 / 45.0 ≈ 0.0889 m/px

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mascara import modo_exportar

ESCALA_M_PX = 4.0 / 45.0   # metros por píxel


def calcular_cinematica(csv_path: str, fps: float = 30.0) -> pd.DataFrame:
    """
    Lee el CSV generado por modo_exportar y calcula:
      - Posición del centroide en píxeles (x, y) y en metros
      - Velocidad instantánea por diferencias finitas hacia adelante
      - Aceleración instantánea (segunda derivada numérica)

    Parámetros
    ----------
    csv_path : ruta al CSV generado por modo_exportar
    fps      : fotogramas por segundo del video original

    Retorna
    -------
    df : DataFrame con las columnas calculadas
    """
    dt = 1.0 / fps

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # Tiempo en segundos para cada frame
    df['tiempo_s'] = df['frame'] * dt

    # ── 2. Posición en metros ─────────────────────────────────────────────────
    # Se utiliza el eje Y del centroide (movimiento a lo largo del carril)
    # y el eje X para desplazamiento lateral.
    # Los valores NaN (frames sin detección) se propagan correctamente.
    df['pos_x_m'] = df['centroide_x'] * ESCALA_M_PX
    df['pos_y_m'] = df['centroide_y'] * ESCALA_M_PX

    # ── 3. Velocidad instantánea — diferencias finitas hacia adelante ─────────
    #
    #   v(t_i) = [ x(t_{i+1}) - x(t_i) ] / dt
    #
    # pandas.diff(periods=-1) calcula el cambio hacia adelante (siguiente - actual).
    # El último frame queda como NaN (no hay t_{i+1}).
    #
    df['vel_x_px_s']  = -df['centroide_x'].diff(periods=-1) / dt
    df['vel_y_px_s']  = -df['centroide_y'].diff(periods=-1) / dt
    df['vel_x_m_s']   = df['vel_x_px_s'] * ESCALA_M_PX
    df['vel_y_m_s']   = df['vel_y_px_s'] * ESCALA_M_PX

    # Magnitud del vector velocidad (rapidez escalar)
    df['rapidez_m_s']  = np.sqrt(df['vel_x_m_s']**2 + df['vel_y_m_s']**2)
    df['rapidez_km_h'] = df['rapidez_m_s'] * 3.6

    # ── 4. Aceleración instantánea — segunda diferencia finita ────────────────
    #
    #   a(t_i) = [ v(t_{i+1}) - v(t_i) ] / dt
    #
    df['acc_x_m_s2']  = -df['vel_x_m_s'].diff(periods=-1) / dt
    df['acc_y_m_s2']  = -df['vel_y_m_s'].diff(periods=-1) / dt
    df['acc_mag_m_s2'] = np.sqrt(df['acc_x_m_s2']**2 + df['acc_y_m_s2']**2)

    return df


def graficar_cinematica(df: pd.DataFrame, fps: float = 30.0,
                        guardar: bool = False,
                        output_path: str = 'cinematica.png'):
    """
    Genera 3 subgráficas apiladas:
      1. Posición (px y m) vs. tiempo
      2. Velocidad (m/s y km/h) vs. tiempo
      3. Aceleración (m/s²) vs. tiempo
    """
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle('Análisis Cinemático del Vehículo', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    t = df['tiempo_s']

    # ── Gráfica 1: Posición ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1_r = ax1.twinx()

    ax1.plot(t, df['centroide_y'], color='steelblue', lw=1.2,
             label='Pos Y (px)')
    ax1_r.plot(t, df['pos_y_m'], color='seagreen', lw=1.2, linestyle='--',
               label='Pos Y (m)')

    ax1.set_ylabel('Posición Y (px)', color='steelblue')
    ax1_r.set_ylabel('Posición Y (m)', color='seagreen')
    ax1.set_title('Posición del centroide vs. tiempo')
    ax1.tick_params(axis='y', colors='steelblue')
    ax1_r.tick_params(axis='y', colors='seagreen')
    ax1.grid(True, alpha=0.3)

    lines1 = ax1.get_lines() + ax1_r.get_lines()
    ax1.legend(lines1, [l.get_label() for l in lines1],
               fontsize=8, loc='upper left')

    # ── Gráfica 2: Velocidad ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2_r = ax2.twinx()

    ax2.plot(t, df['rapidez_m_s'], color='tomato', lw=1.2,
             label='Rapidez (m/s)')
    ax2_r.plot(t, df['rapidez_km_h'], color='darkorange', lw=1.0,
               linestyle='--', label='Rapidez (km/h)')

    ax2.set_ylabel('Velocidad (m/s)', color='tomato')
    ax2_r.set_ylabel('Velocidad (km/h)', color='darkorange')
    ax2.set_title('Velocidad instantánea vs. tiempo  [diferencias finitas hacia adelante]')
    ax2.tick_params(axis='y', colors='tomato')
    ax2_r.tick_params(axis='y', colors='darkorange')
    ax2.axhline(0, color='gray', lw=0.7, linestyle=':')
    ax2.grid(True, alpha=0.3)

    lines2 = ax2.get_lines() + ax2_r.get_lines()
    ax2.legend([l for l in lines2 if l.get_label()[0] != '_'],
               [l.get_label() for l in lines2 if l.get_label()[0] != '_'],
               fontsize=8, loc='upper left')

    # ── Gráfica 3: Aceleración ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    ax3.plot(t, df['acc_y_m_s2'], color='mediumpurple', lw=1.2,
             label='Acel. Y (m/s²)')
    ax3.plot(t, df['acc_mag_m_s2'], color='indigo', lw=0.8,
             linestyle=':', label='|Acel.| (m/s²)')

    ax3.axhline(0, color='gray', lw=0.7, linestyle=':')
    ax3.set_ylabel('Aceleración (m/s²)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_title('Aceleración instantánea vs. tiempo  [segunda derivada numérica]')
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)

    if guardar:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfica guardada → '{output_path}'")
    else:
        plt.show()

    plt.close()


def resumen_cinematico(df: pd.DataFrame):
    """Imprime estadísticas clave del movimiento."""
    print("\n══════ Resumen Cinemático ══════")
    print(f"  Escala usada       : {ESCALA_M_PX:.5f} m/px  "
          f"(45 px = 4 m entre puntos medios del ROI)")
    print(f"  Frames con detección: {df['centroide_y'].notna().sum()} / {len(df)}")
    print(f"  Duración analizada : {df['tiempo_s'].max():.2f} s")
    print(f"\n  -- Posición Y --")
    print(f"  Rango (px)         : {df['centroide_y'].min():.0f} – "
          f"{df['centroide_y'].max():.0f} px")
    print(f"  Rango (m)          : {df['pos_y_m'].min():.2f} – "
          f"{df['pos_y_m'].max():.2f} m")
    print(f"\n  -- Velocidad --")
    print(f"  Vel. máxima        : {df['rapidez_m_s'].max():.2f} m/s  "
          f"({df['rapidez_km_h'].max():.1f} km/h)")
    print(f"  Vel. media         : {df['rapidez_m_s'].mean():.2f} m/s  "
          f"({df['rapidez_km_h'].mean():.1f} km/h)")
    print(f"\n  -- Aceleración --")
    print(f"  Acel. máx. (Y)     : {df['acc_y_m_s2'].abs().max():.2f} m/s²")
    print(f"  Acel. media (|a|)  : {df['acc_mag_m_s2'].mean():.2f} m/s²")
    print("══════════════════════════════\n")


# ══════════════════════════════════════════════════════════════════════════════
# USO — agregar al bloque __main__
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    VIDEO = 'video.mp4'
    CSV   = 'trayectoria_centroide.csv'
    FPS   = 30.0   # <-- ajusta al FPS real de tu video

    # Paso 1: generar CSV con la trayectoria
    modo_exportar(VIDEO, output_csv=CSV)

    # Paso 2: análisis cinemático
    df_cinem = calcular_cinematica(CSV, fps=FPS)
    resumen_cinematico(df_cinem)
    graficar_cinematica(df_cinem, fps=FPS)

    # Opcional: guardar gráfica
    # graficar_cinematica(df_cinem, fps=FPS, guardar=True, output_path='cinematica.png')

    # Opcional: exportar CSV enriquecido con todas las columnas cinemáticas
    # df_cinem.to_csv('cinematica_completa.csv', index=False)