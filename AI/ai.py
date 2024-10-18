import os
import cv2
import mediapipe as mp
import numpy as np
import json

# MediaPipe 설정
mp_pose_video = mp.solutions.pose

# L2 정규화 함수 (벡터의 길이를 1로 만듦)
def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
    return vector / norm

# 키포인트 추출 함수
def extract_keypoints(landmarks, indices):
    return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])

# 비디오 파일에서 키포인트 추출
def extract_video_keypoints(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    with mp_pose_video.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                keypoints = extract_keypoints(results.pose_landmarks.landmark, range(33))
                keypoints_sequence.append(l2_normalize(keypoints))
    
    cap.release()

    # 좌표 데이터를 JSON 파일로 저장
    with open(output_json, 'w') as f:
        json.dump([k.tolist() for k in keypoints_sequence], f)

    print(f"동영상 키포인트가 {output_json}에 저장되었습니다.")

# 실행 코드
if __name__ == "__main__":
    # 동영상 파일 경로 및 JSON 파일 경로 지정
    video_file = 'video/kpop1.mp4'  # 처리할 동영상 파일 경로
    output_json = 'kpop1.json'  # 저장할 JSON 파일 경로

    # 1. 동영상에서 키포인트 추출 및 저장
    extract_video_keypoints(video_file, output_json)
