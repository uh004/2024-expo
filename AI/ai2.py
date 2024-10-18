import os
import cv2
import mediapipe as mp
import numpy as np
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# L2 정규화 함수 (벡터의 길이를 1로 만듦)
def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
    return vector / norm

# 키포인트 추출 함수
def extract_keypoints(landmarks, indices):
    return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])

# 저장된 동영상 키포인트 불러오기
def load_video_keypoints(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} 파일이 존재하지 않습니다.")
        return []
    
    with open(json_file_path, 'r') as f:
        return json.load(f)

# 동영상 재생 및 웹캠 포즈 비교
def play_video_and_compare(video_path, video_keypoints, indices, frame_skip=5):
    cap_video = cv2.VideoCapture(video_path)
    cap_webcam = cv2.VideoCapture(0)  # 웹캠을 연다
    if not cap_webcam.isOpened() or not cap_video.isOpened():
        print("Error: Cannot access webcam or video file.")
        return

    # 비디오 FPS 가져오기
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # FPS에 맞춰 동영상 재생 속도 설정

    # MediaPipe Pose 모델 초기화
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0  # 비교할 프레임을 계산하기 위한 카운터

        while cap_webcam.isOpened() and cap_video.isOpened():
            ret_video, frame_video = cap_video.read()
            ret_webcam, frame_webcam = cap_webcam.read()

            if not ret_video:  # 동영상이 끝나면 반복 종료
                break

            if not ret_webcam:  # 웹캠 입력 오류 처리
                break

            # 웹캠 프레임 처리 (매 frame_skip 프레임마다 비교)
            if frame_count % frame_skip == 0:
                frame_webcam_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
                results_webcam = pose.process(frame_webcam_rgb)

                # 동영상과 웹캠 포즈를 비교
                if results_webcam.pose_landmarks:
                    # 웹캠의 포즈 랜드마크 추출
                    webcam_keypoints = extract_keypoints(results_webcam.pose_landmarks.landmark, indices)
                    webcam_keypoints = l2_normalize(webcam_keypoints)

                    # 동영상의 미리 추출된 포즈와 비교
                    if frame_count < len(video_keypoints):
                        video_keypoints_frame = np.array(video_keypoints[frame_count])

                        # fastdtw를 사용해 두 포즈 간의 거리 계산
                        distance, _ = fastdtw(webcam_keypoints, video_keypoints_frame, dist=euclidean)
                        similarity = 100 - distance  # 유사도는 거리가 적을수록 높다

                        # 유사도 출력
                        print(f"Frame {frame_count}: Similarity = {similarity:.2f}/100")
                    
                    # 웹캠의 포즈 랜드마크를 그리기
                    mp_drawing.draw_landmarks(
                        frame_webcam, 
                        results_webcam.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS
                    )

            # 동영상과 웹캠을 같은 창에 표시하기 위해 크기 맞추기
            frame_webcam = cv2.resize(frame_webcam, (640, 480))
            frame_video = cv2.resize(frame_video, (640, 480))

            # 두 프레임을 나란히 결합
            combined_frame = np.hstack((frame_video, frame_webcam))

            # 결과를 화면에 표시 (왼쪽: 동영상, 오른쪽: 웹캠)
            cv2.imshow('Video and Webcam Pose Comparison', combined_frame)

            # ESC 키를 누르면 종료
            if cv2.waitKey(delay) & 0xFF == 27:
                break

            frame_count += 1  # 다음 프레임으로 넘어가기

    cap_webcam.release()
    cap_video.release()
    cv2.destroyAllWindows()

# 메인 실행 함수
if __name__ == "__main__":
    # 동영상 파일 경로 및 미리 추출된 키포인트 JSON 파일 경로 설정
    video_file = 'video/kpop1.mp4'  # 처리할 동영상 파일 경로
    json_file = 'kpop1.json'  # 미리 추출된 JSON 파일 경로

    # 미리 추출된 동영상 키포인트 데이터를 JSON 파일로부터 불러오기
    video_keypoints = load_video_keypoints(json_file)

    if video_keypoints:
        print(f"{len(video_keypoints)} 프레임의 키포인트가 성공적으로 로드되었습니다.")
        
        # 비교할 랜드마크 인덱스 설정 (모든 관절을 비교하려면 range(33)을 사용)
        landmark_indices = range(33)

        # 동영상 재생 및 웹캠 포즈 비교 함수 실행 (프레임을 5개씩 건너뛰기)
        play_video_and_compare(video_file, video_keypoints, landmark_indices, frame_skip=5)
    else:
        print("키포인트를 불러오지 못했습니다.")
