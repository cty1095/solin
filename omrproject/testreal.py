import cv2
import numpy as np

# -------------------------------
# ✅ 1. 이미지 불러오기
# -------------------------------
image_path = "test image.png"
orig = cv2.imread(image_path)
debug_img = orig.copy()  # 정렬 마크 디버깅용 복사본

# -------------------------------
# ✅ 2. 흑백 + 이진화
# -------------------------------
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# -------------------------------
# ✅ 3. 정렬 마크 검출 + 시각화 (초록 원)
# -------------------------------
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
h, w = orig.shape[:2]
alignment_marks = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    x, y, bw, bh = cv2.boundingRect(cnt)
    cx, cy = x + bw // 2, y + bh // 2
    if 800 < area < 15000 and circularity > 0.2:
        if (cx < w * 0.2 or cx > w * 0.8) and (cy < h * 0.2 or cy > h * 0.8):
            alignment_marks.append((cx, cy))
            cv2.circle(debug_img, (cx, cy), 10, (0, 255, 0), 2)  # 초록 원 표시

# 정렬 마크 정렬
alignment_marks = sorted(alignment_marks, key=lambda p: p[1])
top = sorted(alignment_marks[:2], key=lambda p: p[0])
bottom = sorted(alignment_marks[-2:], key=lambda p: p[0])
sorted_marks = [top[0], top[1], bottom[1], bottom[0]]

# -------------------------------
# ✅ 4. 원근 보정
# -------------------------------
pts_src = np.float32(sorted_marks)
pts_dst = np.float32([[0, 0], [800, 0], [800, 1100], [0, 1100]])
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
aligned = cv2.warpPerspective(orig, M, (800, 1100))
aligned_for_debug = aligned.copy()

# -------------------------------
# ✅ 5. 템플릿 설정
# -------------------------------
start_x = 0.368
dx = 0.122
start_y = 0.195
dy = 0.097

# -------------------------------
# ✅ 6. 채점 + ROI 박스 시각화 (빨간 사각형)
# -------------------------------
gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
_, thresh_aligned = cv2.threshold(gray_aligned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = thresh_aligned.shape
marked_answers = []

for i in range(8):
    max_fill = 0
    selected_idx = -1
    for j in range(5):
        rx = round(start_x + j * dx, 4)
        ry = round(start_y + i * dy, 4)
        cx = int(rx * w)
        cy = int(ry * h)

        # ROI 시각화 (빨간 박스)
        cv2.rectangle(aligned_for_debug, (cx-20, cy-20), (cx+20, cy+20), (0, 0, 255), 1)
        cv2.putText(aligned_for_debug, f"{j+1}", (cx-8, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        roi = thresh_aligned[cy-20:cy+20, cx-20:cx+20]
        filled = cv2.countNonZero(roi)
        if filled > max_fill and filled > 300:
            max_fill = filled
            selected_idx = j
    marked_answers.append(selected_idx + 1)  # 1~5 보기 번호화

# -------------------------------
# ✅ 7. 디버깅용 시각화 출력
# -------------------------------
cv2.imshow("정렬 마크 시각화 (초록 원)", debug_img)
cv2.imshow("보정 후 ROI 박스 시각화 (빨간 박스)", aligned_for_debug)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------
# ✅ 8. 채점 결과 출력
# -------------------------------
print("선택된 보기 (1~5 인덱스):", marked_answers)
