from flask import Flask, Response, render_template, send_file
import cv2
import mediapipe as mp
import numpy as np
from threading import Event
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib
matplotlib.use('Agg')  # 백엔드를 'Agg'로 설정하여 GUI 경고 해결

app = Flask(__name__)

# MediaPipe 설정
mp_pose_webcam = mp.solutions.pose  # 웹캠용 MediaPipe 포즈 추정 모델 설정
mp_pose_video = mp.solutions.pose   # 비디오 파일용 MediaPipe 포즈 추정 모델 설정
mp_drawing = mp.solutions.drawing_utils  # 포즈 랜드마크 그리기 도구

# 비디오 파일 경로 및 오디오 파일 경로 설정
video_file = 'static/video/kpop1.mp4'
audio_file = 'static/audio/kpop1.mp3'

# 프레임 크기를 동일하게 설정
frame_width = 640
frame_height = 480

# 웹캠과 비디오에서 추출한 키포인트들을 저장하는 리스트
keypoints_sequence1 = []  # 웹캠용 키포인트 시퀀스 저장
keypoints_sequence2 = []  # 비디오용 키포인트 시퀀스 저장

# 동기화 이벤트 생성 (웹캠과 비디오가 모두 준비될 때까지 대기)
first_frame_ready_event = Event()  # 첫 번째 프레임 준비 여부 확인
stop_event = Event()  # 동영상이 끝나면 웹캠도 종료하기 위한 이벤트

# 비디오의 FPS 가져오기
cap2 = cv2.VideoCapture(video_file)
if not cap2.isOpened():
    print("Error: Cannot open video file.")
else:
    # FPS 가져오기
    fps = cap2.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

cap2.release()

# L2 정규화 함수
def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
    return vector / norm

# 키포인트 추출 함수
def extract_keypoints(landmarks, indices):
    return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])

# 웹캠 프레임 생성 및 키포인트 저장
def generate_webcam_frames():
    cap1 = cv2.VideoCapture(0)  # 웹캠

    if not cap1.isOpened():
        print("Error: Cannot access webcam.")
        return

    first_frame = None

    with mp_pose_webcam.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose1:
        while cap1.isOpened() and not stop_event.is_set():  # stop_event가 설정되면 종료
            ret1, frame1 = cap1.read()

            if not ret1:
                break

            # 첫 번째 프레임을 저장하고 준비가 되었음을 알림
            if first_frame is None:
                first_frame = frame1
                first_frame_ready_event.set()  # 첫 번째 프레임이 준비되었음을 알림
                first_frame_ready_event.wait()  # 비디오 첫 프레임 준비를 대기
            
            # 프레임 좌우 반전 (1은 좌우반전 의미)
            frame1 = cv2.flip(frame1, 1)
            
            # 프레임 크기 조정
            frame1 = cv2.resize(frame1, (frame_width, frame_height))

            # MediaPipe 포즈 추정
            frame1.flags.writeable = False
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results1 = pose1.process(frame1_rgb)
            frame1.flags.writeable = True

            # 포즈 관절 그리기
            if results1.pose_landmarks:
                mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose_webcam.POSE_CONNECTIONS)

                keypoints = extract_keypoints(results1.pose_landmarks.landmark, range(33))
                keypoints_sequence1.append(l2_normalize(keypoints))  # 웹캠 키포인트 저장

            ret, buffer = cv2.imencode('.jpg', frame1)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap1.release()

# 비디오 파일 프레임 생성 및 키포인트 저장
def generate_video_frames():
    cap2 = cv2.VideoCapture(video_file)  # 비디오 파일

    if not cap2.isOpened():
        print("Error: Cannot open video file.")
        return

    first_frame = None

    with mp_pose_video.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose2:
        while cap2.isOpened():
            ret2, frame2 = cap2.read()

            if not ret2:
                stop_event.set()  # 비디오가 끝나면 stop_event 설정
                break

            # 첫 번째 프레임을 저장하고 준비가 되었음을 알림
            if first_frame is None:
                first_frame = frame2
                first_frame_ready_event.set()  # 첫 번째 프레임이 준비되었음을 알림
                first_frame_ready_event.wait()  # 웹캠 첫 프레임 준비를 대기

            # 프레임 크기 조정
            frame2 = cv2.resize(frame2, (frame_width, frame_height))

            # MediaPipe 포즈 추정
            frame2.flags.writeable = False
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results2 = pose2.process(frame2_rgb)
            frame2.flags.writeable = True

            # 포즈 관절 그리기
            if results2.pose_landmarks:
                mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose_video.POSE_CONNECTIONS)

                keypoints = extract_keypoints(results2.pose_landmarks.landmark, range(33))
                keypoints_sequence2.append(l2_normalize(keypoints))  # 비디오 키포인트 저장

            ret, buffer = cv2.imencode('.jpg', frame2)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap2.release()

# 실시간 스트림 제공 (포즈 비교 없이)
def stream_webcam():
    webcam_gen = generate_webcam_frames()  # 웹캠 프레임 생성기

    while True:
        try:
            frame = next(webcam_gen)  # 프레임을 바이트 형식으로 반환
            yield frame
        except StopIteration:
            break

def stream_video():
    video_gen = generate_video_frames()  # 비디오 프레임 생성기

    while True:
        try:
            frame = next(video_gen)  # 프레임을 바이트 형식으로 반환
            yield frame
        except StopIteration:
            break

# 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# 포즈 유사도 비교 함수 (춤이 끝난 후)
def compare_pose_dtw():
    if len(keypoints_sequence1) == 0 or len(keypoints_sequence2) == 0:
        return []

    aligned_keypoints1 = []  # 첫 번째 비디오의 정렬된 키포인트 시퀀스 저장
    aligned_keypoints2 = []  # 두 번째 비디오의 정렬된 키포인트 시퀀스 저장

    similarity_list = []  # 유사도를 저장할 리스트

    # 각 관절 그룹별로 DTW 적용
    for key in range(33):  # 각 관절 인덱스에 대해 처리
        sequence1 = np.array([kp[key] for kp in keypoints_sequence1])  # 첫 번째 비디오의 키포인트 시퀀스
        sequence2 = np.array([kp[key] for kp in keypoints_sequence2])  # 두 번째 비디오의 키포인트 시퀀스

        # DTW를 사용한 정렬
        distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
        aligned_sequence1 = [sequence1[idx1] for idx1, idx2 in path]
        aligned_sequence2 = [sequence2[idx2] for idx1, idx2 in path]

        aligned_keypoints1.append(aligned_sequence1)
        aligned_keypoints2.append(aligned_sequence2)

    # 평균 유사도 계산
    for i in range(len(aligned_keypoints1[0])):  # 시퀀스 길이에 대해 처리
        similarities = []
        for j in range(len(aligned_keypoints1)):
            if i < len(aligned_keypoints1[j]) and i < len(aligned_keypoints2[j]):
                similarity = cosine_similarity(aligned_keypoints1[j][i], aligned_keypoints2[j][i])
                similarities.append(similarity)
        if similarities:
            average_similarity = np.mean(similarities) * 100  # 평균 유사도를 100점 만점으로 변환
            similarity_list.append(average_similarity)
        else:
            similarity_list.append(0)  # 유사도가 없으면 0점으로 처리

    return similarity_list

# 유사도 결과를 그래프로 시각화하고 이미지를 반환
@app.route('/graph')
def visualize_similarity():
    similarity_list = compare_pose_dtw()

    if not similarity_list:
        return "No similarity data available."

    # 시간 축 값 생성 (FPS를 기반으로 시간 계산)
    time_values = np.arange(0, len(similarity_list)) * (1 / fps)  # FPS로 시간을 계산

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_values, similarity_list, label='Pose Similarity Score after DTW', color='blue', linewidth=2.5, marker='o')

    # 기준선 그리기 (90점 기준)
    ax.axhline(y=90, color='red', linestyle='--', label='90 Score Threshold', linewidth=2)

    # x축, y축 라벨 및 제목 설정
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pose Similarity Score (0-100)', fontsize=14, fontweight='bold')
    ax.set_title('Pose Similarity Score Over Time', fontsize=16, fontweight='bold')

    # 그리드 추가 (격자 선)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 축 눈금 스타일 조정
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 범례 스타일 조정
    ax.legend(loc='upper left', fontsize=12)

    # 이미지를 버퍼에 저장
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')

# Flask 경로 설정
@app.route('/')
def index():
    return render_template('index.html', audio_file=audio_file)

@app.route('/webcam_feed')
def webcam_feed():
    return Response(stream_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/compare')
def compare():
    return "Similarity comparison complete. Go to /graph to see the results."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
