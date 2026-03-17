import cv2
import numpy as np
 
# ── Parámetros ideales (fijos) ─────────────────────────────────────────────────
CONTRASTE   = 1.4
GAMMA       = 0.1
BLUR_VAL    = 0
NUEVO_ANCHO = 680
NUEVO_ALTO  = 480
 
# ── Parámetros morfológicos (ajusta según lo que encontraste) ──────────────────
BG_METODO    = 'MOG2'
MORFO_OP     = 3          # 0=Erosion 1=Dilatacion 2=Apertura 3=Cierre
KERNEL_FORMA = 1          # 0=Rect 1=Elipse 2=Cruz
KERNEL_SIZE  = 7
ITERACIONES  = 2
AREA_MIN     = 800        # área mínima en px² para considerar un contorno como vehículo
 
# ── Polígono ROI ───────────────────────────────────────────────────────────────
puntos_roi = np.array([
    [238,  63],
    [276,  58],
    [326, 130],
    [405, 470],
    [272, 470],
    [281, 130],
], dtype=np.int32)
 
mascara_roi = np.zeros((NUEVO_ALTO, NUEVO_ANCHO), dtype=np.uint8)
cv2.fillPoly(mascara_roi, [puntos_roi], 255)
x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(puntos_roi)
 
# ── Tabla gamma ────────────────────────────────────────────────────────────────
tabla_gamma = np.array([
    np.clip(((i / 255.0) ** (1.0 / GAMMA)) * 255, 0, 255)
    for i in range(256)
], dtype=np.uint8)
 
# ── Sustractores de fondo ──────────────────────────────────────────────────────
sustractor_MOG2 = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)
sustractor_KNN = cv2.createBackgroundSubtractorKNN(
    history=500, dist2Threshold=400.0, detectShadows=True
)
sustractor = sustractor_MOG2 if BG_METODO == 'MOG2' else sustractor_KNN
 
 
# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES BASE
# ══════════════════════════════════════════════════════════════════════════════
 
def preprocesar_frame(frame):
    frame_r = cv2.resize(frame, (NUEVO_ANCHO, NUEVO_ALTO))
    gris = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    gris = np.clip(gris.astype(np.float32) * CONTRASTE, 0, 255).astype(np.uint8)
    gris = cv2.LUT(gris, tabla_gamma)
    return frame_r, gris
 
 
def obtener_mascara_fg(gris):
    """BG subtraction + eliminar sombras + restricción al ROI."""
    fg = sustractor.apply(gris)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.bitwise_and(fg, fg, mask=mascara_roi)
    return fg
 
 
def aplicar_morfologia(mascara):
    """Limpieza morfológica con los parámetros fijos encontrados."""
    shapes = {0: cv2.MORPH_RECT, 1: cv2.MORPH_ELLIPSE, 2: cv2.MORPH_CROSS}
    ops    = {0: cv2.MORPH_ERODE, 1: cv2.MORPH_DILATE, 2: cv2.MORPH_OPEN, 3: cv2.MORPH_CLOSE}
    k = KERNEL_SIZE if KERNEL_SIZE % 2 == 1 else KERNEL_SIZE + 1
    kernel = cv2.getStructuringElement(shapes[KERNEL_FORMA], (k, k))
    return cv2.morphologyEx(mascara, ops[MORFO_OP], kernel, iterations=ITERACIONES)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE CONTORNOS Y CENTROIDE
# ══════════════════════════════════════════════════════════════════════════════
 
def detectar_vehiculo(mascara_limpia):
    """
    Encuentra todos los contornos externos en la máscara limpia.
    Selecciona el contorno de MAYOR área como el vehículo principal.
    Calcula el centroide usando momentos de imagen (cv2.moments).
 
    Retorna:
        contorno_principal : np.ndarray o None
        centroide          : (cx, cy) en coordenadas del frame completo, o None
        todos_contornos    : lista de todos los contornos detectados
        area_principal     : float, área del contorno principal
    """
    # ── 1. Encontrar contornos externos ───────────────────────────────────────
    # cv2.RETR_EXTERNAL  : solo contornos externos (ignora huecos internos)
    # cv2.CHAIN_APPROX_SIMPLE : comprime segmentos rectos, ahorra memoria
    contornos, _ = cv2.findContours(
        mascara_limpia,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
 
    # Filtrar contornos por área mínima
    contornos_validos = [c for c in contornos if cv2.contourArea(c) >= AREA_MIN]
 
    if not contornos_validos:
        return None, None, [], 0.0
 
    # ── 2. Seleccionar el contorno principal (mayor área = vehículo) ──────────
    contorno_principal = max(contornos_validos, key=cv2.contourArea)
    area_principal     = cv2.contourArea(contorno_principal)
 
    # ── 3. Calcular centroide con momentos de imagen ──────────────────────────
    #
    # cv2.moments(contorno) devuelve un diccionario con los momentos espaciales:
    #   m00 = área del contorno
    #   m10 = suma de coordenadas x ponderadas
    #   m01 = suma de coordenadas y ponderadas
    #
    # Centroide: cx = m10 / m00
    #            cy = m01 / m00
    #
    momentos = cv2.moments(contorno_principal)
 
    centroide = None
    if momentos['m00'] != 0:
        cx = int(momentos['m10'] / momentos['m00'])
        cy = int(momentos['m01'] / momentos['m00'])
        centroide = (cx, cy)
 
    return contorno_principal, centroide, contornos_validos, area_principal
 
 
def dibujar_resultado(frame_color, contorno_principal, centroide,
                      todos_contornos, area_principal, rastro):
    """
    Dibuja sobre el frame:
      - Todos los contornos válidos en verde claro
      - Contorno principal del vehículo en verde brillante
      - Bounding box del contorno principal en azul
      - Cruz + círculo en el centroide
      - Rastro de posiciones anteriores del centroide
      - Información numérica en pantalla
    """
    display = frame_color.copy()
 
    # Polígono ROI de referencia
    cv2.polylines(display, [puntos_roi], isClosed=True,
                  color=(0, 200, 255), thickness=1)
 
    if todos_contornos:
        # Todos los contornos válidos (verde tenue)
        cv2.drawContours(display, todos_contornos, -1, (0, 180, 80), 1)
 
    if contorno_principal is not None:
        # Contorno principal del vehículo (verde brillante, grueso)
        cv2.drawContours(display, [contorno_principal], -1, (0, 255, 0), 2)
 
        # Bounding box en azul
        bx, by, bw, bh = cv2.boundingRect(contorno_principal)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh),
                      (255, 100, 0), 1)
 
        # Info de área junto al bbox
        cv2.putText(display, f'Area: {int(area_principal)} px2',
                    (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 200, 0), 1)
 
    if centroide is not None:
        cx, cy = centroide
 
        # Rastro de trayectoria: puntos anteriores con degradado de color
        for i, punto in enumerate(rastro):
            alpha = int(255 * (i + 1) / len(rastro))   # más brillante = más reciente
            cv2.circle(display, punto, 2, (0, alpha, 255 - alpha), -1)
 
        # Cruz del centroide
        tam_cruz = 12
        cv2.line(display,
                 (cx - tam_cruz, cy), (cx + tam_cruz, cy),
                 (0, 0, 255), 2)
        cv2.line(display,
                 (cx, cy - tam_cruz), (cx, cy + tam_cruz),
                 (0, 0, 255), 2)
 
        # Círculo exterior del centroide
        cv2.circle(display, (cx, cy), 6, (0, 0, 255), 2)
 
        # Coordenadas del centroide en pantalla
        cv2.putText(display, f'Centroide: ({cx}, {cy})',
                    (cx + 14, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
 
    # Panel de información superior
    cv2.putText(display,
                f'BG: {BG_METODO}  |  Contornos validos: {len(todos_contornos)}',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
 
    estado      = f'Vehiculo detectado  Area={int(area_principal)}px2' \
                  if contorno_principal is not None else 'Sin vehiculo en ROI'
    color_estado = (0, 255, 100) if contorno_principal is not None else (0, 100, 255)
    cv2.putText(display, estado,
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_estado, 2)
 
    return display
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — Visualización interactiva
# ══════════════════════════════════════════════════════════════════════════════
def modo_visualizacion(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir '{video_path}'")
        return
 
    pausar    = False
    frame     = None
    rastro    = []
    MAX_RASTRO = 40
 
    print("Teclas: [P] Pausar  [R] Reiniciar  [Esc] Salir")
 
    while True:
        if not pausar:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                rastro.clear()
                ret, frame = cap.read()
                if not ret:
                    break
 
        if frame is None:
            break
 
        frame_r, gris  = preprocesar_frame(frame)
        mascara_fg     = obtener_mascara_fg(gris)
        mascara_limpia = aplicar_morfologia(mascara_fg)
 
        contorno, centroide, todos, area = detectar_vehiculo(mascara_limpia)
 
        if centroide is not None:
            rastro.append(centroide)
            if len(rastro) > MAX_RASTRO:
                rastro.pop(0)
 
        display           = dibujar_resultado(frame_r, contorno, centroide, todos, area, rastro)
        mascara_recortada = mascara_limpia[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]
 
        cv2.imshow('Contorno y Centroide', display)
        cv2.imshow('Mascara limpia ROI',   mascara_recortada)
 
        key = cv2.waitKey(33) & 0xFF
        if key == 27:
            break
        elif key == ord('p'):
            pausar = not pausar
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            rastro.clear()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — Exportar video anotado + CSV con trayectoria del centroide
# ══════════════════════════════════════════════════════════════════════════════
def modo_exportar(video_path: str,
                  output_video: str = 'video_contorno.mp4',
                  output_csv:   str = 'trayectoria_centroide.csv'):
    import csv
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir '{video_path}'")
        return
 
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video, fourcc, fps,
                             (NUEVO_ANCHO, NUEVO_ALTO), isColor=True)
 
    rastro = []
 
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Encabezado del CSV
        writer.writerow(['frame', 'centroide_x', 'centroide_y',
                         'area_px2', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
 
        print(f"Exportando '{output_video}' y '{output_csv}' ...")
 
        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
 
            frame_r, gris  = preprocesar_frame(frame)
            mascara_fg     = obtener_mascara_fg(gris)
            mascara_limpia = aplicar_morfologia(mascara_fg)
 
            contorno, centroide, todos, area = detectar_vehiculo(mascara_limpia)
 
            if centroide is not None:
                rastro.append(centroide)
                if len(rastro) > 40:
                    rastro.pop(0)
                bx, by, bw, bh = cv2.boundingRect(contorno)
                writer.writerow([idx, centroide[0], centroide[1],
                                  int(area), bx, by, bw, bh])
            else:
                # Frame sin detección → fila vacía para mantener índice
                writer.writerow([idx, '', '', '', '', '', '', ''])
 
            display = dibujar_resultado(frame_r, contorno, centroide,
                                        todos, area, rastro)
            out.write(display)
 
            if idx % 50 == 0:
                print(f"  {idx}/{total}  ({idx / total * 100:.1f}%)")
 
    cap.release()
    out.release()
    print(f"Listo.\n  Video  → '{output_video}'\n  CSV    → '{output_csv}'")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    VIDEO = 'video.mp4'   # <-- cambia al nombre/ruta de tu video
 
    modo_visualizacion(VIDEO)
 
    # Para exportar video anotado + CSV con la trayectoria del centroide:
    # modo_exportar(VIDEO)