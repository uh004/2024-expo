# 필요한 라이브러리
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pygame

# 미디어파이프 도구 설정
mp_drawing1 = mp.solutions.drawing_utils  # 포즈 랜드마크 그리기 도구
mp_drawing_styles1 = mp.solutions.drawing_styles  # 포즈 랜드마크 스타일 도구
mp_pose1 = mp.solutions.pose  # 첫 번째 포즈 추정 모델

mp_drawing2 = mp.solutions.drawing_utils  # 두 번째 포즈 랜드마크 그리기 도구
mp_drawing_styles2 = mp.solutions.drawing_styles  # 두 번째 포즈 랜드마크 스타일 도구
mp_pose2 = mp.solutions.pose  # 두 번째 포즈 추정 모델

# 웹캠 디바이스 번호 및 비디오 파일 경로 설정
cam_file = 0  # 웹캠의 디바이스 번호 (0번은 기본 카메라)
video_file = 'video/kpop1.mp4'  # 비교할 비디오 파일 경로
audio_file = 'audio/kpop1.mp3'  # 재생할 오디오 파일 경로

# 두 개의 캡처 객체 생성
cap1 = cv2.VideoCapture(cam_file)  # 웹캠 캡처 객체
cap2 = cv2.VideoCapture(video_file)  # 비디오 파일 캡처 객체

# 웹캠 또는 비디오 파일이 열리지 않으면 오류 메시지 출력
if not cap1.isOpened() or not cap2.isOpened():
    print("하나 이상의 입력을 열 수 없습니다.")
    exit()

# 비디오의 프레임 레이트 가져오기
fps = cap2.get(cv2.CAP_PROP_FPS)  # 비디오 파일의 프레임 레이트를 가져옴

# 유사도를 계산할 때 임계값 설정 (예: 0.1 이하의 차이는 무시)
SIMILARITY_THRESHOLD = 0.01

# 코사인 유사도 계산 함수 수정
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)  # 벡터 내적
    norm_v1 = np.linalg.norm(v1)  # 첫 번째 벡터의 크기 계산
    norm_v2 = np.linalg.norm(v2)  # 두 번째 벡터의 크기 계산
    similarity = dot_product / (norm_v1 * norm_v2)  # 코사인 유사도 계산
    # 유사도가 임계값 이하이면 낮은 유사도로 간주
    if similarity > (1 - SIMILARITY_THRESHOLD):
        similarity = 0  # 임계값 이하 차이는 무시
    return similarity

def detailed_joint_analysis(keypoints1, keypoints2, indices):
    # 개별 관절의 유사도 계산
    detailed_similarities = []
    for index in indices:
        v1, v2 = keypoints1[index], keypoints2[index]
        similarity = cosine_similarity(v1, v2)
        detailed_similarities.append(similarity)
    return np.mean(detailed_similarities)

def overall_motion_analysis(keypoints1, keypoints2, indices):
    # 전체 동작 흐름의 유사도 계산
    overall_vectors = []
    for index in indices:
        vector1 = np.array([keypoints1[index]['x'], keypoints1[index]['y']])
        vector2 = np.array([keypoints2[index]['x'], keypoints2[index]['y']])
        overall_vectors.append(cosine_similarity(vector1, vector2))
    return np.mean(overall_vectors)


# 프레임 크기 조정 함수
def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]  # 이미지의 종횡비 계산
    return cv2.resize(image, (width, int(width / aspect_ratio)))  # 주어진 너비에 맞춰 크기 조정

# L2 정규화 함수
def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)  # L2 norm 계산
    return vector / norm  # 각 좌표를 L2 norm으로 나누기

# 프레임 스킵 설정
frame_skip = 2  # 프레임을 건너뛰는 설정 (매 2번째 프레임만 처리)
frame_count = 0  # 현재 프레임 카운트

# 포즈 유사도 값을 저장할 리스트
similarity_list = []  # 유사도를 저장할 리스트 초기화

# 비디오의 총 길이(초) 계산
total_duration = cap2.get(cv2.CAP_PROP_FRAME_COUNT) / fps  # 동영상의 전체 시간을 계산

# MediaPipe 포즈 추정기 설정 및 프레임 처리 루프
with mp_pose1.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose1, \
    mp_pose2.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose2:
    
    keypoints_sequence1 = []  # 첫 번째 비디오의 키포인트 시퀀스 저장
    keypoints_sequence2 = []  # 두 번째 비디오의 키포인트 시퀀스 저장

    # pygame 초기화 및 오디오 파일 로드
    pygame.mixer.init()  # pygame 초기화
    pygame.mixer.music.load(audio_file)  # 오디오 파일 로드
    pygame.mixer.music.play()  # 오디오 재생 시작
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, image1 = cap1.read() # 웹캠 프레임 읽기
        ret2, image2 = cap2.read() # 동영상 프레임 읽기

        if not ret1 or not ret2:
            print("카메라를 찾을 수 없습니다.")
            break
        
        frame_count += 1  # 프레임 카운트 증가
        if frame_count % frame_skip != 0:  # 프레임 스킵 적용
            continue
        
        image1 = resize_image(image1, 800)  # 웹캠 이미지 크기 조정
        image2 = resize_image(image2, 800)  # 비디오 이미지 크기 조정
        
        image1.flags.writeable = False  # 이미지 메모리를 읽기 전용으로 설정
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        results1 = pose1.process(image1)  # 첫 번째 포즈 추정 수행

        image1.flags.writeable = True  # 이미지 메모리를 쓰기 가능으로 설정
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)  # RGB를 BGR로 다시 변환

        # 포즈 랜드마크의 기본 스타일 설정
        default_landmarks_style = mp_drawing1.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        default_connections_style = mp_drawing1.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)

        # 0-9번 점들에 대한 특별한 스타일 설정
        special_landmarks_style = mp_drawing1.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=20)

        # 기본 스타일로 모든 랜드마크를 그림
        # mp_drawing1.draw_landmarks(
        #     image1, 
        #     results1.pose_landmarks, 
        #     mp_pose1.POSE_CONNECTIONS, 
        #     landmark_drawing_spec=default_landmarks_style,
        #     connection_drawing_spec=default_connections_style
        # )
        # 11,13,15,23,25,27,29,31,12,14,16,24,26,28,30,32번 점에 대한 특별한 스타일로 그림
        if results1.pose_landmarks:
            for idx, landmark in enumerate(results1.pose_landmarks.landmark):
                if idx in [11,13,15,23,25,27,29,31,12,14,16,24,26,28,30,32]:
                    x = int(landmark.x * image1.shape[1])
                    y = int(landmark.y * image1.shape[0])
                    cv2.circle(image1, (x, y), special_landmarks_style.circle_radius, special_landmarks_style.color, special_landmarks_style.thickness)
        
        # 이거 주석하면 속도 돌아옴
        image2.flags.writeable = False  # 두 번째 이미지 메모리를 읽기 전용으로 설정
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        results2 = pose2.process(image2)  # 두 번째 포즈 추정 수행

        image2.flags.writeable = True  # 두 번째 이미지 메모리를 쓰기 가능으로 설정
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)  # RGB를 BGR로 다시 변환
        # mp_drawing2.draw_landmarks(image2, results2.pose_landmarks, mp_pose2.POSE_CONNECTIONS, 
        #                         landmark_drawing_spec=mp_drawing_styles2.get_default_pose_landmarks_style())
        
        height = min(image1.shape[0], image2.shape[0])  # 두 이미지의 높이 중 작은 값 선택
        width = min(image1.shape[1], image2.shape[1])  # 두 이미지의 너비 중 작은 값 선택
        
        frame1 = cv2.resize(image1, (width, height))  # 첫 번째 이미지를 동일 크기로 조정
        frame2 = cv2.resize(image2, (width, height))  # 두 번째 이미지를 동일 크기로 조정

        frame1 = cv2.flip(frame1, 1)  # 첫 번째 이미지를 좌우 반전

        combined_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)  # 두 이미지를 담을 빈 프레임 생성
        combined_frame[0:height, 0:width] = frame2  # 왼쪽에 두 번째 이미지 배치
        combined_frame[0:height, width:width*2] = frame1  # 오른쪽에 첫 번째 이미지 배치

        # 현재 동영상의 거꾸로 된 시간을 계산
        elapsed_time = frame_count / fps  # 경과 시간을 초 단위로 계산
        remaining_time = total_duration - elapsed_time  # 남은 시간을 계산

        # 시간을 텍스트로 변환
        time_text = f'Time Remaining: {remaining_time:.0f} sec'

        # 텍스트 위치 설정
        text_position = (800, 25)  # 프레임 아래쪽에 텍스트 위치 설정

        # 텍스트 스타일 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2

        # 텍스트를 합쳐진 프레임에 그리기
        cv2.putText(combined_frame, time_text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        cv2.imshow('combined_frame', combined_frame)  # 합쳐진 이미지를 화면에 표시
        
        # 포즈 랜드마크가 둘 다 존재할 때
        if results1.pose_landmarks and results2.pose_landmarks:
            # 키포인트 추출 함수
            def extract_keypoints(landmarks, indices):
                keypoints = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])  # 선택된 인덱스의 랜드마크를 추출
                keypoints = l2_normalize(keypoints)  # L2 정규화 적용
                return keypoints

            left_arm_indices = [11, 13, 15]  # 왼쪽 팔의 랜드마크 인덱스
            left_leg_indices = [23, 25, 27]  # 왼쪽 다리의 랜드마크 인덱스
            # left_foot_indices = [27, 29, 31]  # 왼쪽 발의 랜드마크 인덱스
            right_arm_indices = [12, 14, 16]  # 오른쪽 팔의 랜드마크 인덱스
            right_leg_indices = [24, 26, 28]  # 오른쪽 다리의 랜드마크 인덱스
            # right_foot_indices = [28, 30, 32]  # 오른쪽 발의 랜드마크 인덱스

            keypoints1 = {
                'left_arm': extract_keypoints(results1.pose_landmarks.landmark, left_arm_indices),
                'left_leg': extract_keypoints(results1.pose_landmarks.landmark, left_leg_indices),
                # 'left_foot': extract_keypoints(results1.pose_landmarks.landmark, left_foot_indices),
                'right_arm': extract_keypoints(results1.pose_landmarks.landmark, right_arm_indices),
                'right_leg': extract_keypoints(results1.pose_landmarks.landmark, right_leg_indices),
                # 'right_foot': extract_keypoints(results1.pose_landmarks.landmark, right_foot_indices)
            }

            keypoints2 = {
                'left_arm': extract_keypoints(results2.pose_landmarks.landmark, left_arm_indices),
                'left_leg': extract_keypoints(results2.pose_landmarks.landmark, left_leg_indices),
                # 'left_foot': extract_keypoints(results2.pose_landmarks.landmark, left_foot_indices),
                'right_arm': extract_keypoints(results2.pose_landmarks.landmark, right_arm_indices),
                'right_leg': extract_keypoints(results2.pose_landmarks.landmark, right_leg_indices),
                # 'right_foot': extract_keypoints(results2.pose_landmarks.landmark, right_foot_indices)
            }

            keypoints_sequence1.append(keypoints1)  # 첫 번째 비디오의 키포인트 시퀀스 추가
            keypoints_sequence2.append(keypoints2)  # 두 번째 비디오의 키포인트 시퀀스 추가

        # 'ESC' 키가 눌리면 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 캡처 객체 및 창 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()

# 각 관절 그룹별로 DTW 적용
aligned_keypoints1 = []  # 첫 번째 비디오의 정렬된 키포인트 시퀀스 저장
aligned_keypoints2 = []  # 두 번째 비디오의 정렬된 키포인트 시퀀스 저장

for key in keypoints_sequence1[0].keys():
    sequence1 = np.array([kp[key].flatten() for kp in keypoints_sequence1])  # 첫 번째 비디오의 키포인트 시퀀스
    sequence2 = np.array([kp[key].flatten() for kp in keypoints_sequence2])  # 두 번째 비디오의 키포인트 시퀀스
    
    distance, path = fastdtw(sequence1, sequence2, dist=euclidean)  # DTW를 사용한 정렬
    aligned_sequence1 = [sequence1[idx1] for idx1, idx2 in path]  # 첫 번째 비디오의 정렬된 시퀀스
    aligned_sequence2 = [sequence2[idx2] for idx1, idx2 in path]  # 두 번째 비디오의 정렬된 시퀀스
    
    aligned_keypoints1.append(aligned_sequence1)  # 정렬된 첫 번째 시퀀스 추가
    aligned_keypoints2.append(aligned_sequence2)  # 정렬된 두 번째 시퀀스 추가

# 평균 유사도 계산 부분도 유사도 차이를 더 명확히 반영하도록 수정:
for i in range(len(aligned_keypoints1[0])):
    similarities = []
    for j in range(len(aligned_keypoints1)):
        if i < len(aligned_keypoints1[j]) and i < len(aligned_keypoints2[j]):
            similarity = cosine_similarity(aligned_keypoints1[j][i], aligned_keypoints2[j][i])  # 코사인 유사도 계산
            similarities.append(similarity)
    if similarities:
        average_similarity = np.mean(similarities) * 100  # 평균 유사도를 100점 만점으로 변환
        similarity_list.append(average_similarity)
    else:
        similarity_list.append(0)  # 유사도가 없으면 0점으로 처리

# 시간 축 값 생성
time_values = np.arange(0, len(similarity_list)) * (frame_skip / fps)

# 유사도를 0-100 점수로 변환
score_list = [similarity for similarity in similarity_list]

# 그래프 크기 설정
plt.figure(figsize=(12, 6))

# 유사도 점수 그래프 그리기
plt.plot(time_values, similarity_list, label='Pose Similarity Score after DTW', color='blue', linewidth=2.5, marker='o')

# 기준선 그리기
plt.axhline(y=90, color='red', linestyle='--', label='90 Score Threshold', linewidth=2)

# x축, y축 라벨 및 제목 설정
plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
plt.ylabel('Pose Similarity Score (0-100)', fontsize=14, fontweight='bold')
plt.title('Pose Similarity Score Over Time', fontsize=16, fontweight='bold')

# 그리드 추가 (격자 선)
plt.grid(True, linestyle='--', alpha=0.6)

# 축 눈금 스타일 조정
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 범례 스타일 조정
plt.legend(loc='upper left', fontsize=12)

# 그래프 출력
plt.show()