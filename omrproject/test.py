import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# ✅ 1. 마킹된 OMR 이미지 불러오기
# -------------------------------
image_path = "test image.png"  # 학생 답안 이미지
orig = cv2.imread(image_path)

cv2.imshow("original image", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()


# -------------------------------
# ✅ 2. 흑백 변환 + 이진화 (마킹 검출용)
# -------------------------------
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow("gray image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("threshed image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()



# -------------------------------
# ✅ 3. 정렬 마크(4개) 검출
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

    # 정렬 마크 조건: 충분한 면적 + 모서리에 위치
    if 800 < area < 15000 and circularity > 0.2:
        if (cx < w * 0.2 or cx > w * 0.8) and (cy < h * 0.2 or cy > h * 0.8):
            alignment_marks.append((cx, cy))

# 상단 2개, 하단 2개 → 좌상 / 우상 / 우하 / 좌하 순서로 정렬
alignment_marks = sorted(alignment_marks, key=lambda p: p[1])
top = sorted(alignment_marks[:2], key=lambda p: p[0])
bottom = sorted(alignment_marks[-2:], key=lambda p: p[0])
sorted_marks = [top[0], top[1], bottom[1], bottom[0]]

# -------------------------------
# ✅ 4. 원근 보정 (정렬 마크 기준)
# -------------------------------
pts_src = np.float32(sorted_marks)
pts_dst = np.float32([[0, 0], [800, 0], [800, 1100], [0, 1100]])  # 출력 크기
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
aligned = cv2.warpPerspective(orig, M, (800, 1100))

# -------------------------------
# ✅ 5. 템플릿 좌표 기준 설정 (상대좌표)
# -------------------------------
start_x = 0.368  # 보기 1번의 X 시작 위치
dx = 0.122       # 보기 간의 간격
start_y = 0.195  # 문제 1번의 Y 시작 위치
dy = 0.097       # 문제 간의 간격

# -------------------------------
# ✅ 6. 보정된 이미지에서 마킹 채점
# -------------------------------
gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
_, thresh_aligned = cv2.threshold(gray_aligned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = thresh_aligned.shape
marked_answers = []

for i in range(8):  # 8문제
    max_fill = 0
    selected_idx = -1
    for j in range(5):  # 5보기
        # 상대좌표 → 절대픽셀 위치 변환
        rx = round(start_x + j * dx, 4)
        ry = round(start_y + i * dy, 4)
        cx = int(rx * w)
        cy = int(ry * h)

        # 선택된 영역(ROI) 자르기 (40x40)
        roi = thresh_aligned[cy-20:cy+20, cx-20:cx+20]
        filled = cv2.countNonZero(roi)

        # 채움 픽셀이 많고 일정 기준(300)을 넘으면 마킹된 것으로 간주
        if filled > max_fill and filled > 300:
            max_fill = filled
            selected_idx = j

    marked_answers.append(selected_idx+1)

# -------------------------------
# ✅ 7. 채점 결과 출력
# -------------------------------
# result = []
# for i in range(len(marked_answers)):
#     a=marked_answers[i] +1
#     result.append(a)
print(marked_answers)

