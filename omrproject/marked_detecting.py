'''
찐 최종 
1.원본 omr로 템플릿 완성
2.중복, 미마킹 검사
'''


# 1. 라이브러리 불러오기
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# 2. 이미지 불러오기
image_path = "marked omr test image.png"
image = cv2.imread(image_path)
orig = image.copy()

# 3. 이진화 처리 (Threshold)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. 윤곽선 찾기
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 정렬 마크 필터링 (위치 기반: 외곽 20%)
h, w = image.shape[:2]
position_filtered_candidates = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    cx, cy = x + w_box // 2, y + h_box // 2
    if 1000 < area < 10000 and 0.3 < circularity < 0.8:
        if (cx < w * 0.2 or cx > w * 0.8) and (cy < h * 0.2 or cy > h * 0.8):
            position_filtered_candidates.append((cx, cy))

# 6. 정렬 마크 좌표 정렬 함수
def sort_alignment_points(points):
    points = sorted(points, key=lambda p: p[1])  # y 기준 정렬
    top = sorted(points[:2], key=lambda p: p[0])
    bottom = sorted(points[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[1], bottom[0]]  # TL, TR, BR, BL

# 7. 이미지 보정 수행
if len(position_filtered_candidates) == 4:
    sorted_pts = sort_alignment_points(position_filtered_candidates)
    w_out, h_out = 800, 1100  # 보정된 출력 크기
    template_pts = np.float32([[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]])
    detected_pts = np.float32(sorted_pts)

    M = cv2.getPerspectiveTransform(detected_pts, template_pts)
    warped = cv2.warpPerspective(orig, M, (w_out, h_out))

    # 보정된 이미지 시각화
    plt.figure(figsize=(6, 10))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Perspective-Corrected OMR Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("정렬 마크가 정확히 4개 감지되지 않아 보정을 수행할 수 없습니다.")

# 8. 템플릿 좌표 생성 (상대 좌표 기준)
bubble_template = {
    "alignment_marks": {
        "top_left": [0.0, 0.0],
        "top_right": [1.0, 0.0],
        "bottom_right": [1.0, 1.0],
        "bottom_left": [0.0, 1.0]
    },
    "bubbles": {}
}

start_y = 0.1  # 첫 문제 y 시작 위치
start_x = 0.15 # 첫 보기 x 시작 위치
dy = 0.07      # 문제 간 y 간격
dx = 0.13      # 보기 간 x 간격

for q in range(10):
    question_id = str(q + 1)
    bubble_template["bubbles"][question_id] = []
    y = start_y + q * dy
    for i in range(5):
        x = start_x + i * dx
        bubble_template["bubbles"][question_id].append([round(x, 3), round(y, 3)])

# 9. 템플릿 JSON 일부 출력
template_json = json.dumps(bubble_template, indent=2)
print(template_json[:1000])

with open("omr_template.json", "w") as f:
    json.dump(bubble_template, f, indent=2)