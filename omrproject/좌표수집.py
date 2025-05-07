import cv2

# 1. 이미지 불러오기
image_path = "marked omr student.png"  # 정렬(Perspective 보정)된 이미지 파일
image = cv2.imread(image_path)
h, w = image.shape[:2]

# 2. 클릭 이벤트 함수
clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rel_x = round(x / w, 4)
        rel_y = round(y / h, 4)
        clicked_points.append((rel_x, rel_y))
        print(f"클릭한 위치: (x={x}, y={y}) → 상대좌표: ({rel_x}, {rel_y})")
        # 시각적으로도 표시
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("OMR 클릭 - 상대좌표 출력", image)

# 3. 창 띄우기
cv2.namedWindow("OMR 클릭 - 상대좌표 출력", cv2.WINDOW_NORMAL)
cv2.imshow("OMR 클릭 - 상대좌표 출력", image)
cv2.setMouseCallback("OMR 클릭 - 상대좌표 출력", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. 결과 좌표 리스트
print("📌 수집된 상대좌표 목록:")
for i, pt in enumerate(clicked_points, 1):
    print(f"{i}. {pt}")
