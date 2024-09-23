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
video_file = 'video/child_tocatoca.mp4'  # 비교할 비디오 파일 경로
audio_file = 'sound/child_tocatoca.mp3'  # 재생할 오디오 파일 경로

# 두 개의 캡처 객체 생성
cap1 = cv2.VideoCapture(cam_file)  # 웹캠 캡처 객체
cap2 = cv2.VideoCapture(video_file)  # 비디오 파일 캡처 객체

# 웹캠 또는 비디오 파일이 열리지 않으면 오류 메시지 출력
if not cap1.isOpened() or not cap2.isOpened():
    print("하나 이상의 입력을 열 수 없습니다.")
    exit()

# 비디오의 프레임 레이트 가져오기
fps = cap2.get(cv2.CAP_PROP_FPS)  # 비디오 파일의 프레임 레이트를 가져옴

# 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)  # 벡터 내적
    norm_v1 = np.linalg.norm(v1)  # 첫 번째 벡터의 크기 계산
    norm_v2 = np.linalg.norm(v2)  # 두 번째 벡터의 크기 계산
    return dot_product / (norm_v1 * norm_v2)  # 코사인 유사도 계산

# 프레임 크기 조정 함수
def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]  # 이미지의 종횡비 계산
    return cv2.resize(image, (width, int(width / aspect_ratio)))  # 주어진 너비에 맞춰 크기 조정

# 프레임 스킵 설정
frame_skip = 2  # 프레임을 건너뛰는 설정 (매 2번째 프레임만 처리)
frame_count = 0  # 현재 프레임 카운트

# 포즈 유사도 값을 저장할 리스트
similarity_list = []  # 유사도를 저장할 리스트 초기화

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
        mp_drawing1.draw_landmarks(
            image1, 
            results1.pose_landmarks, 
            mp_pose1.POSE_CONNECTIONS, 
            landmark_drawing_spec=default_landmarks_style,
            connection_drawing_spec=default_connections_style
        )

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
        mp_drawing2.draw_landmarks(image2, results2.pose_landmarks, mp_pose2.POSE_CONNECTIONS, 
                                landmark_drawing_spec=mp_drawing_styles2.get_default_pose_landmarks_style())
        
        height = min(image1.shape[0], image2.shape[0])  # 두 이미지의 높이 중 작은 값 선택
        width = min(image1.shape[1], image2.shape[1])  # 두 이미지의 너비 중 작은 값 선택
        
        frame1 = cv2.resize(image1, (width, height))  # 첫 번째 이미지를 동일 크기로 조정
        frame2 = cv2.resize(image2, (width, height))  # 두 번째 이미지를 동일 크기로 조정

        frame1 = cv2.flip(frame1, 1)  # 첫 번째 이미지를 좌우 반전

        combined_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)  # 두 이미지를 담을 빈 프레임 생성
        combined_frame[0:height, 0:width] = frame2  # 왼쪽에 두 번째 이미지 배치
        combined_frame[0:height, width:width*2] = frame1  # 오른쪽에 첫 번째 이미지 배치

        cv2.imshow('combined_frame', combined_frame)  # 합쳐진 이미지를 화면에 표시
        
        # 포즈 랜드마크가 둘 다 존재할 때
        if results1.pose_landmarks and results2.pose_landmarks:
            def extract_keypoints(landmarks, indices):
                return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])  # 선택된 인덱스의 랜드마크를 추출

            left_arm_indices = [11, 13, 15]  # 왼쪽 팔의 랜드마크 인덱스
            left_leg_indices = [23, 25, 27]  # 왼쪽 다리의 랜드마크 인덱스
            left_foot_indices = [27, 29, 31]  # 왼쪽 발의 랜드마크 인덱스
            right_arm_indices = [12, 14, 16]  # 오른쪽 팔의 랜드마크 인덱스
            right_leg_indices = [24, 26, 28]  # 오른쪽 다리의 랜드마크 인덱스
            right_foot_indices = [28, 30, 32]  # 오른쪽 발의 랜드마크 인덱스

            keypoints1 = {
                'left_arm': extract_keypoints(results1.pose_landmarks.landmark, left_arm_indices),
                'left_leg': extract_keypoints(results1.pose_landmarks.landmark, left_leg_indices),
                'left_foot': extract_keypoints(results1.pose_landmarks.landmark, left_foot_indices),
                'right_arm': extract_keypoints(results1.pose_landmarks.landmark, right_arm_indices),
                'right_leg': extract_keypoints(results1.pose_landmarks.landmark, right_leg_indices),
                'right_foot': extract_keypoints(results1.pose_landmarks.landmark, right_foot_indices)
            }

            keypoints2 = {
                'left_arm': extract_keypoints(results2.pose_landmarks.landmark, left_arm_indices),
                'left_leg': extract_keypoints(results2.pose_landmarks.landmark, left_leg_indices),
                'left_foot': extract_keypoints(results2.pose_landmarks.landmark, left_foot_indices),
                'right_arm': extract_keypoints(results2.pose_landmarks.landmark, right_arm_indices),
                'right_leg': extract_keypoints(results2.pose_landmarks.landmark, right_leg_indices),
                'right_foot': extract_keypoints(results2.pose_landmarks.landmark, right_foot_indices)
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

# 평균 유사도 계산
for i in range(len(aligned_keypoints1[0])):
    similarities = []
    for j in range(len(aligned_keypoints1)):
        # 인덱스가 범위 내에 있는지 확인
        if i < len(aligned_keypoints1[j]) and i < len(aligned_keypoints2[j]):
            similarity = cosine_similarity(aligned_keypoints1[j][i], aligned_keypoints2[j][i])  # 코사인 유사도 계산
            similarities.append(similarity)
        else:
            pass
    
    if similarities:
        average_similarity = np.mean(similarities)  # 평균 유사도 계산
        similarity_list.append(average_similarity)  # 유사도 리스트에 추가
    else:
        print("No similarities calculated for this frame.")

# 시간 축 값 생성
time_values = np.arange(0, len(similarity_list)) * (frame_skip / fps)

# 유사도 시각화
plt.figure(figsize=(10, 5))
plt.plot(time_values, similarity_list, label='Cosine Similarity after DTW')  # 유사도 그래프 그리기
plt.axhline(y=0.9, color='r', linestyle='--', label='0.9 Threshold')  # 기준선을 그리기
plt.xlabel('Time (seconds)')  # x축 라벨
plt.ylabel('Cosine Similarity')  # y축 라벨
plt.title('Pose Similarity Over Time')  # 그래프 제목
plt.legend()  # 범례 표시
plt.show()  # 그래프 출력
