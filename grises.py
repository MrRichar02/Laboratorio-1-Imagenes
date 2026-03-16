import cv2
import numpy as np
 
# Variables globales
pausar_video = False
mostrar_pixel = False
x, y = 0, 0
 
# Función de callback de mouse
def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = _x, _y
        mostrar_pixel = True
    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video
 
# Cargar el video
video_path = 'video.mp4'  # Cambia esto al camino de tu video
cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
 
if not cap.isOpened():
    print("Error: no se pudo abrir video")
else:
    print("Video abierto correctamente")
 
cv2.namedWindow('Video Gris')
cv2.setMouseCallback('Video Gris', mouse_callback)
 
nuevo_alto = 480
nuevo_ancho = 680
 
# ── Polígono ROI ───────────────────────────────────────────────────────────────
# 6 puntos tomados sobre el frame redimensionado (680x480), en orden de contorno:
#   1(238,63) → 2(300,58) → 3(355,130) → 4(405,470) → 5(272,470) → 6(261,130)
puntos_roi = np.array([
    [238,  63],   # 1 - superior izquierdo
    [300,  58],   # 2 - superior derecho
    [355, 130],   # 3 - medio derecho
    [405, 470],   # 4 - inferior derecho
    [272, 470],   # 5 - inferior izquierdo
    [281, 130],   # 6 - medio izquierdo
], dtype=np.int32)
 
# Pre-calcular la máscara ROI una sola vez (no cambia entre frames)
mascara_roi = np.zeros((nuevo_alto, nuevo_ancho), dtype=np.uint8)
cv2.fillPoly(mascara_roi, [puntos_roi], 255)
 
# Bounding box del polígono para recortar la ventana ROI
x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(puntos_roi)
 
def nothing(x):
    pass
 
# Trackbars para escala de grises
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Gris Min",  "Trackbars", 0,   255, nothing)
cv2.createTrackbar("Gris Max",  "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Contraste", "Trackbars", 10,  30,  nothing)  # ÷10 → 0.0–3.0
cv2.createTrackbar("Gamma x10", "Trackbars", 10,  50,  nothing)  # ÷10 → 0.1–5.0
cv2.createTrackbar("Blur",      "Trackbars", 0,   20,  nothing)  # kernel gaussiano
 
while True:
    if not pausar_video:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frame2 = np.copy(frame)
 
    if mostrar_pixel:
        pixel_color = frame[y, x]
        frame2 = np.copy(frame)
        cv2.putText(frame2, f'Pos ({x},{y}) BGR: {pixel_color}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
    # Redimensionar
    frame3 = cv2.resize(frame2, (nuevo_ancho, nuevo_alto))
 
    # ── Escala de grises ───────────────────────────────────────────────────────
    gris = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
 
    # ── Leer trackbars ─────────────────────────────────────────────────────────
    gris_min  = cv2.getTrackbarPos("Gris Min",  "Trackbars")
    gris_max  = cv2.getTrackbarPos("Gris Max",  "Trackbars")
    contraste = cv2.getTrackbarPos("Contraste", "Trackbars") / 10.0
    gamma     = max(cv2.getTrackbarPos("Gamma x10", "Trackbars") / 10.0, 0.1)
    blur_val  = cv2.getTrackbarPos("Blur",      "Trackbars")
 
    # ── Blur ───────────────────────────────────────────────────────────────────
    if blur_val > 0:
        k = blur_val * 2 + 1
        gris_proc = cv2.GaussianBlur(gris, (k, k), 0)
    else:
        gris_proc = gris.copy()
 
    # ── Contraste ──────────────────────────────────────────────────────────────
    gris_proc = np.clip(gris_proc.astype(np.float32) * contraste, 0, 255).astype(np.uint8)
 
    # ── Gamma (tabla LUT) ──────────────────────────────────────────────────────
    tabla_gamma = np.array([
        np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    gris_proc = cv2.LUT(gris_proc, tabla_gamma)
 
    # ── Máscara por umbral aplicada solo al ROI ────────────────────────────────
    mascara_full = cv2.inRange(gris_proc, gris_min, gris_max)
    # Combinar la máscara de umbral con la máscara del polígono
    mascara_roi_umbral = cv2.bitwise_and(mascara_full, mascara_full, mask=mascara_roi)
    # Recortar al bounding box del polígono
    mascara_recortada = mascara_roi_umbral[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]
 
    # ── Ventana principal: gris con polígono dibujado ──────────────────────────
    gris_display = cv2.cvtColor(gris_proc, cv2.COLOR_GRAY2BGR)
    cv2.polylines(gris_display, [puntos_roi], isClosed=True, color=(0, 255, 0), thickness=2)
    # Marcar los 6 puntos con número
    for i, (px, py) in enumerate(puntos_roi):
        cv2.circle(gris_display, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(gris_display, str(i + 1), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(gris_display,
                f'Contraste:{contraste:.1f}  Gamma:{gamma:.1f}  Blur:{blur_val}',
                (10, nuevo_alto - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    cv2.putText(gris_display,
                f'Umbral: [{gris_min} - {gris_max}]',
                (10, nuevo_alto - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
 
    # ── Ventana ROI: área dentro del polígono con imagen en gris ──────────────
    roi_aplicado = cv2.bitwise_and(gris_proc, gris_proc, mask=mascara_roi)
    roi_recortado = roi_aplicado[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]
 
    # ── Mostrar ventanas ───────────────────────────────────────────────────────
    cv2.imshow('Video Gris',    gris_display)
    cv2.imshow('Video Mascara', mascara_recortada)   # máscara solo dentro del ROI
    cv2.imshow('Video ROI',     roi_recortado)
 
    key = cv2.waitKey(33) & 0xFF
    if key == 27:           # Esc → salir
        break
    elif key == ord('r'):   # R   → reiniciar video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        mostrar_pixel = False
 
cap.release()
cv2.destroyAllWindows()
