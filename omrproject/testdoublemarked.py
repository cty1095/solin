'''
찐 최종 
1.원본 omr로 템플릿 완성
2.중복, 미마킹 검사


앞으로 해야될것
정답이랑 채점 비교 -> 점수 출력(csv파일로 정답키 입력예정)
코드 함수화
'''



import cv2
import numpy as np

#이미지 출력할때 사이즈 조정
def resize_for_debug(img, scale=0.5):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))



# 이미지 불러오기
image_path = "wndqhr.png"
orig = cv2.imread(image_path)



if orig is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")


#오리진 이미지 출력(아무키 입력)
debug_img = resize_for_debug(orig, scale=0.5)
cv2.imshow("original image", debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 흑백 변환 + 이진화

gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#흑백 이미지 출력
debug_img_gray = resize_for_debug(gray, scale=0.5)
cv2.imshow("gray image", debug_img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이진화 이미지 출력
debug_img_thresh = resize_for_debug(thresh, scale=0.5)
cv2.imshow("threshed image", debug_img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 정렬 마크 검출

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

alignment_marks = sorted(alignment_marks, key=lambda p: p[1])
top = sorted(alignment_marks[:2], key=lambda p: p[0])
bottom = sorted(alignment_marks[-2:], key=lambda p: p[0])
sorted_marks = [top[0], top[1], bottom[1], bottom[0]]


#원근 보정

pts_src = np.float32(sorted_marks)
pts_dst = np.float32([[0, 0], [800, 0], [800, 1100], [0, 1100]])
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
aligned = cv2.warpPerspective(orig, M, (800, 1100))


#템플릿 상대좌표

start_x = 0.368
dx = 0.122
start_y = 0.195
dy = 0.097


#채점 + 중복 마킹 감지

gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
_, thresh_aligned = cv2.threshold(gray_aligned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = thresh_aligned.shape
marked_answers = []


for i in range(8):  # 문제 수
    filled_counts = []
    for j in range(5):  # 보기 수
        rx = round(start_x + j * dx, 4)
        ry = round(start_y + i * dy, 4)
        cx = int(rx * w)
        cy = int(ry * h)

        roi = thresh_aligned[cy-20:cy+20, cx-20:cx+20]
        filled = cv2.countNonZero(roi)
        filled_counts.append(filled)

    # 중복 마킹 판단
    threshold = 700
    marked_indices = [idx for idx, val in enumerate(filled_counts) if val > threshold]
    
    if len(marked_indices) == 0:
        marked_answers.append("미마킹")  
    elif len(marked_indices) == 1:
        marked_answers.append(marked_indices[0] + 1)  
    else:
        marked_answers.append("중복")  


# 결과

print("채점 결과:")
for idx, result in enumerate(marked_answers, 1):
    print(f"문제 {idx}: {result}")
