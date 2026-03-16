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
#   1(238,63) → 2(300,58) → 3(355,130) → 4(405,470) → 5(272,470) → 6(281,130)
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

# Trackbars HSV + Contraste + Gamma + Blur
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Min",     "Trackbars", 0,   179, nothing)
cv2.createTrackbar("H Max",     "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min",     "Trackbars", 0,   255, nothing)
cv2.createTrackbar("S Max",     "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min",     "Trackbars", 0,   255, nothing)
cv2.createTrackbar("V Max",     "Trackbars", 255, 255, nothing)
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

    # ── Leer trackbars ─────────────────────────────────────────────────────────
    h_min     = cv2.getTrackbarPos("H Min",     "Trackbars")
    h_max     = cv2.getTrackbarPos("H Max",     "Trackbars")
    s_min     = cv2.getTrackbarPos("S Min",     "Trackbars")
    s_max     = cv2.getTrackbarPos("S Max",     "Trackbars")
    v_min     = cv2.getTrackbarPos("V Min",     "Trackbars")
    v_max     = cv2.getTrackbarPos("V Max",     "Trackbars")
    contraste = cv2.getTrackbarPos("Contraste", "Trackbars") / 10.0
    gamma     = max(cv2.getTrackbarPos("Gamma x10", "Trackbars") / 10.0, 0.1)
    blur_val  = cv2.getTrackbarPos("Blur",      "Trackbars")

    # ── Blur ───────────────────────────────────────────────────────────────────
    if blur_val > 0:
        k = blur_val * 2 + 1
        frame_proc = cv2.GaussianBlur(frame3, (k, k), 0)
    else:
        frame_proc = frame3.copy()

    # ── Contraste ──────────────────────────────────────────────────────────────
    frame_proc = np.clip(frame_proc.astype(np.float32) * contraste, 0, 255).astype(np.uint8)

    # ── Gamma (tabla LUT) ──────────────────────────────────────────────────────
    tabla_gamma = np.array([
        np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    frame_proc = cv2.LUT(frame_proc, tabla_gamma)

    # ── Conversión a HSV ───────────────────────────────────────────────────────
    hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)

    # ── Máscara HSV por umbral ─────────────────────────────────────────────────
    bajo = np.array([h_min, s_min, v_min])
    alto = np.array([h_max, s_max, v_max])
    mascara_full = cv2.inRange(hsv, bajo, alto)

    # Combinar máscara de umbral con la máscara del polígono y recortar
    mascara_roi_umbral = cv2.bitwise_and(mascara_full, mascara_full, mask=mascara_roi)
    mascara_recortada  = mascara_roi_umbral[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]

    # ── Ventana principal: HSV con polígono dibujado ───────────────────────────
    hsv_display = hsv.copy()
    cv2.polylines(hsv_display, [puntos_roi], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (px, py) in enumerate(puntos_roi):
        cv2.circle(hsv_display, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(hsv_display, str(i + 1), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(hsv_display,
                f'Contraste:{contraste:.1f}  Gamma:{gamma:.1f}  Blur:{blur_val}',
                (10, nuevo_alto - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    cv2.putText(hsv_display,
                f'H:[{h_min}-{h_max}]  S:[{s_min}-{s_max}]  V:[{v_min}-{v_max}]',
                (10, nuevo_alto - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    # ── Ventana ROI: área dentro del polígono con imagen HSV ──────────────────
    roi_aplicado = cv2.bitwise_and(hsv, hsv, mask=mascara_roi)
    roi_recortado = roi_aplicado[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]

    # ── Mostrar ventanas ───────────────────────────────────────────────────────
    cv2.imshow('Video Gris',    hsv_display)
    cv2.imshow('Video Mascara', mascara_recortada)
    cv2.imshow('Video ROI',     roi_recortado)

    key = cv2.waitKey(33) & 0xFF
    if key == 27:           # Esc → salir
        break
    elif key == ord('r'):   # R   → reiniciar video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        mostrar_pixel = False

cap.release()
cv2.destroyAllWindows()
