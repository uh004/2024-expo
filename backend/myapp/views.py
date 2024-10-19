from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Post, Comment
from .models import Player
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import StreamingHttpResponse, FileResponse, Http404
from django.conf import settings
import os
from django.templatetags.static import static

# Create your views here.


def start(request):
  return render(request, 'start.html')


def save_nickname(request):
    if request.method == "POST":
        nickname = request.POST.get('nickname')  # 닉네임 가져오기
        gender = request.POST.get('gender')  # 성별 가져오기 (M 또는 F)

        # 데이터베이스에 저장
        player, created = Player.objects.get_or_create(nickname=nickname, defaults={'gender': gender})

        if not created:
            # 기존 플레이어라면 성별을 업데이트 (선택을 다시 변경한 경우)
            player.gender = gender
            player.save()
            
        request.session['nickname'] = nickname
        return redirect('mypage')  # mypage로 리다이렉트

    return render(request, 'start.html')


def main(request):
  return render(request, 'main.html')

def gamerule(request):
  return render(request, 'gamerule.html')

def rank(request):
  return render(request, 'rank.html')

def comunity(request):
  posts = Post.objects.all() # 데이터베이스에서 모든 게시글 가져오기 ( Django 서버 측에서 데이터 가져오는곳임)
  return render(request, 'comunity.html', {'posts': posts})

def write(request):
  return render(request, 'write.html')

def submit_post(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        video = request.FILES.get('video')  # 파일 업로드 처리
        nickname = request.session.get('nickname')
        
        if nickname:
          author = get_object_or_404(Player, nickname=nickname)  # 작성자 가져오기
          post = Post.objects.create(title=title, content=content, video=video, author=author)
          return redirect('comunity')  # 게시글 저장 후 커뮤니티로 리다이렉트
        
        
    return render(request, 'write.html')

# 게시물 상세 보기 및 댓글 조회
def get_post_detail(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    comments = post.comments.all()
    return JsonResponse({
        'id': post.id,
        'title': post.title,
        'content': post.content,
        'author': post.author.nickname,
        'video_url': post.video.url if post.video else None,
        'comments': [
            {'author': comment.author.nickname, 'content': comment.content, 'created_at': comment.created_at.strftime('%Y-%m-%d')}
            for comment in comments
        ]
    })

# 댓글 추가 기능
def submit_comment(request, post_id):
    if request.method == 'POST':
        content = request.POST.get('content')
        nickname = request.session.get('nickname')

        if content and nickname:
            try:
                author = get_object_or_404(Player, nickname=nickname)
                post = get_object_or_404(Post, id=post_id)
                comment = Comment.objects.create(post=post, author=author, content=content)

                return JsonResponse({
                    'status': 'success', 
                    'author': comment.author.nickname, 
                    'content': comment.content,
                    'created_at': comment.created_at.strftime("%Y-%m-%d %H:%M")
                })
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


def mypage(request):
  nickname = request.session.get('nickname')  # 세션 또는 다른 방식으로 닉네임 전달
  return render(request, 'mypage.html', {'nickname': nickname})

def choice(request):
  return render(request, 'choice.html')

def kpop(request):
  return render(request, 'kpop.html')

def kpopcard1(request):
  return render(request, 'kpopcard1.html')

def kpopcard2(request):
  return render(request, 'kpopcard2.html')

def kpopcard3(request):
  return render(request, 'kpopcard3.html')

def kpopcard1_start(request):
  return render(request, 'kpopcard1_start.html')

def kpopcard2_start(request):
  return render(request, 'kpopcard2_start.html')

def kpopcard3_start(request):
  return render(request, 'kpopcard3_start.html')

def shots(request):
  return render(request, 'shots.html')

def shotscard1(request):
  return render(request, 'shotscard1.html')

def shotscard2(request):
  return render(request, 'shotscard2.html')

def shotscard3(request):
  return render(request, 'shotscard3.html')

def challenge(request):
  return render(request, 'challenge.html')

def challengecard1(request):
  return render(request, 'challengecard1.html')

def challengecard2(request):
  return render(request, 'challengecard2.html')

def challengecard3(request):
  return render(request, 'challengecard3.html')


# views.py

import os
import cv2
import mediapipe as mp
import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from threading import Event
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from django.conf import settings
from django.views.decorators.cache import never_cache
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as mcolors

# MediaPipe 설정
mp_pose_webcam = mp.solutions.pose
mp_pose_video = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 비디오 파일 경로 설정
video_file = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'video', 'kpop1.mp4')


# 전역적으로 키포인트 저장을 위한 리스트
keypoints_sequence1 = []
keypoints_sequence2 = []

# 동기화 이벤트 생성 (두 스트림이 독립적으로 실행되도록 하기 위해)
first_frame_ready_event = Event()
stop_event = Event()

# 리셋 함수 - 각 세션마다 키포인트와 이벤트 초기화
def reset_events():
    first_frame_ready_event.clear()
    stop_event.clear()
    keypoints_sequence1.clear()
    keypoints_sequence2.clear()

# L2 정규화 함수 (벡터의 길이를 1로 만듦)
def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
    return vector / norm

# 키포인트 추출 함수
def extract_keypoints(landmarks, indices):
    return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])

# 무지개 색상 배열 (랜덤으로 적용)
rainbow_colors = [
    (255, 0, 0),    # 빨간색
    (255, 165, 0),  # 주황색
    (255, 255, 0),  # 노란색
    (0, 255, 0),    # 초록색
    (0, 0, 255),    # 파란색
    (75, 0, 130),   # 남색
    (238, 130, 238) # 보라색
]

# 색상 변경 속도를 위한 변수
color_change_speed = 0.05  # 한 프레임마다 색상이 5%씩 변화 (값을 줄이면 더 부드럽게 변함)

# 현재 색상과 목표 색상을 저장하는 변수
current_landmark_color = np.array([255, 0, 0], dtype=np.float32)  # 초기 색상 (빨간색)
target_landmark_color = np.array(random.choice(rainbow_colors), dtype=np.float32)

current_connection_color = np.array([255, 0, 0], dtype=np.float32)
target_connection_color = np.array(random.choice(rainbow_colors), dtype=np.float32)

# 랜드마크 스타일 설정 (커다란 동그라미와 두꺼운 선)
landmark_style = mp_drawing.DrawingSpec(color=current_landmark_color, thickness=2, circle_radius=2)
connection_style = mp_drawing.DrawingSpec(color=current_connection_color, thickness=2)

# 색상을 서서히 변화시키는 함수
def smooth_color_transition(current_color, target_color, speed):
    return current_color + (target_color - current_color) * speed

# 랜드마크에 대한 가중치 설정
# 팔, 다리, 발의 각 랜드마크에 다른 가중치를 부여
landmark_weights = np.ones(33)  # 기본 가중치는 1
left_arm_indices = [11, 13, 15]  # 왼쪽 팔
right_arm_indices = [12, 14, 16]  # 오른쪽 팔
left_leg_indices = [23, 25, 27]  # 왼쪽 다리
right_leg_indices = [24, 26, 28]  # 오른쪽 다리
left_foot_indices = [27, 29, 31]  # 왼쪽 발
right_foot_indices = [28, 30, 32]  # 오른쪽 발

# 팔에 가중치 부여
landmark_weights[left_arm_indices] = 1.5
landmark_weights[right_arm_indices] = 1.5
# 다리 가중치 부여
landmark_weights[left_leg_indices] = 1.5
landmark_weights[right_leg_indices] = 1.5
# 발에 가중치 부여
landmark_weights[left_foot_indices] = 1.2
landmark_weights[right_foot_indices] = 1.2

# 코사인 유사도 계산 함수 (가중치 적용)
def cosine_similarity_weighted(v1, v2, weights):
    dot_product = np.dot(v1 * weights, v2 * weights)
    norm_v1 = np.linalg.norm(v1 * weights)
    norm_v2 = np.linalg.norm(v2 * weights)
    return dot_product / (norm_v1 * norm_v2)

# 웹캠 프레임 생성 및 포즈 감지 (아이들이 좋아할 스타일 적용)
def generate_webcam_frames():
    cap1 = cv2.VideoCapture(0)  # 웹캠 초기화
    if not cap1.isOpened():
        print("Error: Cannot access webcam.")
        return

    global current_landmark_color, target_landmark_color
    global current_connection_color, target_connection_color

    # MediaPipe 포즈 감지기 초기화
    with mp_pose_webcam.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose1:
        while cap1.isOpened() and not stop_event.is_set():
            ret1, frame1 = cap1.read()  # 웹캠에서 프레임 읽기
            if not ret1:
                break

            # 프레임 좌우 반전 및 크기 조정
            frame1 = cv2.flip(frame1, 1)
            frame1 = cv2.resize(frame1, (640, 480))

            # 포즈 감지를 위한 전처리
            frame1.flags.writeable = False
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results1 = pose1.process(frame1_rgb)
            frame1.flags.writeable = True

            # 색상 서서히 변화
            current_landmark_color = smooth_color_transition(current_landmark_color, target_landmark_color, color_change_speed)
            current_connection_color = smooth_color_transition(current_connection_color, target_connection_color, color_change_speed)

            # 색상이 충분히 변했을 때 새로운 목표 색상 설정
            if np.linalg.norm(target_landmark_color - current_landmark_color) < 10:
                target_landmark_color = np.array(random.choice(rainbow_colors), dtype=np.float32)
            if np.linalg.norm(target_connection_color - current_connection_color) < 10:
                target_connection_color = np.array(random.choice(rainbow_colors), dtype=np.float32)

            # 포즈 랜드마크가 감지되면
            if results1.pose_landmarks:
                # 랜드마크를 더 귀엽고 재미있게 그리기 (서서히 색상 변화 적용)
                mp_drawing.draw_landmarks(
                    frame1, 
                    results1.pose_landmarks, 
                    mp_pose_webcam.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=tuple(map(int, current_landmark_color)), thickness=4, circle_radius=6),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=tuple(map(int, current_connection_color)), thickness=5)
                )
                
                # 키포인트 추출 및 정규화하여 저장
                keypoints = extract_keypoints(results1.pose_landmarks.landmark, range(33))
                keypoints_sequence1.append(l2_normalize(keypoints))

            # 프레임을 인코딩하여 웹 브라우저로 전송
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap1.release()

# 비디오 프레임 생성 및 포즈 감지
def generate_video_frames():
    cap2 = cv2.VideoCapture(video_file)
    fps = cap2.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps

    if not cap2.isOpened():
        print("Error: Cannot open video file.")
        return

    with mp_pose_video.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose2:
        while cap2.isOpened():
            start_time = time.time()
            ret2, frame2 = cap2.read()
            if not ret2:
                stop_event.set()
                break

            frame2 = cv2.resize(frame2, (640, 480))
            frame2.flags.writeable = False
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results2 = pose2.process(frame2_rgb)
            frame2.flags.writeable = True

            if results2.pose_landmarks:
                keypoints = extract_keypoints(results2.pose_landmarks.landmark, range(33))
                keypoints_sequence2.append(l2_normalize(keypoints))

            ret, buffer = cv2.imencode('.jpg', frame2)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            processing_time = time.time() - start_time
            if processing_time < delay:
                time.sleep(delay - processing_time)

    cap2.release()

def get_video_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return 30  # 기본 FPS 값으로 30을 반환 (비디오가 열리지 않으면 기본값 사용)
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS 가져오기
    cap.release()
    return fps

def cosine_similarity_weighted(v1, v2, weights):
    """
    가중치 적용한 코사인 유사도 계산 함수.
    :param v1: 첫 번째 랜드마크의 3D 벡터 (x, y, z)
    :param v2: 두 번째 랜드마크의 3D 벡터 (x, y, z)
    :param weights: 각 랜드마크에 대한 가중치 배열 (크기 33)
    :return: 가중치가 적용된 코사인 유사도
    """
    weight = weights  # 가중치가 하나의 랜드마크에 적용됨

    # 각 랜드마크의 좌표에 동일한 가중치를 곱한 후 코사인 유사도 계산
    dot_product = np.dot(v1 * weight, v2 * weight)
    norm_v1 = np.linalg.norm(v1 * weight)
    norm_v2 = np.linalg.norm(v2 * weight)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # 벡터 중 하나가 0이면 유사도는 0으로 처리
    
    return dot_product / (norm_v1 * norm_v2)


def compare_pose_dtw():
    """
    웹캠 키포인트 시퀀스와 비디오 키포인트 시퀀스를 비교하여 유사도 계산.
    DTW (동적 시간 왜곡)을 사용하여 두 시퀀스를 정렬하고, 가중치 적용 코사인 유사도로 비교.
    :return: 각 프레임별 평균 유사도 리스트
    """
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
        _, path = fastdtw(sequence1, sequence2, dist=euclidean)
        aligned_sequence1 = [sequence1[idx1] for idx1, _ in path]  # 첫 번째 시퀀스 정렬
        aligned_sequence2 = [sequence2[idx2] for _, idx2 in path]  # 두 번째 시퀀스 정렬

        aligned_keypoints1.append(aligned_sequence1)
        aligned_keypoints2.append(aligned_sequence2)

    # 평균 유사도 계산 (가중치 적용)
    for i in range(len(aligned_keypoints1[0])):  # 시퀀스 길이에 대해 처리
        similarities = []
        for j in range(len(aligned_keypoints1)):
            if i < len(aligned_keypoints1[j]) and i < len(aligned_keypoints2[j]):
                # 각 랜드마크에 대한 가중치를 적용하여 유사도 계산
                similarity = cosine_similarity_weighted(
                    aligned_keypoints1[j][i],
                    aligned_keypoints2[j][i],
                    landmark_weights[j]
                )
                similarities.append(similarity)
        if similarities:
            average_similarity = np.mean(similarities) * 100  # 평균 유사도를 100점 만점으로 변환
            similarity_list.append(average_similarity)
        else:
            similarity_list.append(0)  # 유사도가 없으면 0점으로 처리

    return similarity_list


# 그래프 생성 함수
def graph(request):
    similarity_list = compare_pose_dtw()

    if not similarity_list:
        return HttpResponse("No similarity data available.")

    # 평균 유사도 계산
    average_similarity = np.mean(similarity_list) if similarity_list else 0

    # 비디오의 FPS를 가져옴
    fps = get_video_fps(video_file)

    # FPS에 기반한 시간 축 생성
    time_values = np.arange(0, len(similarity_list)) * (1 / fps)

    # 그라데이션 색상 설정 (단순한 컬러맵으로 대체)
    cmap = plt.get_cmap('viridis')  # 'viridis' 그라데이션 색상 사용
    norm = plt.Normalize(min(similarity_list), max(similarity_list))

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(14, 8))

    # 밝은 배경으로 설정하여 선과 포인트가 더 명확하게 보이도록 함
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9f9f9')

    # 굵은 선과 큰 포인트 마커 스타일 적용
    for i in range(len(similarity_list)):
        ax.plot(
            time_values[i:i+2], similarity_list[i:i+2], color=cmap(norm(similarity_list[i])), linewidth=4
        )
        ax.scatter(time_values[i], similarity_list[i], color=cmap(norm(similarity_list[i])),
                   edgecolor='black', s=150, zorder=5, marker='o')  # 동그란 마커

    # 기준선 추가 (90점 기준)
    ax.axhline(y=90, color='red', linestyle='--', label='90 Score Threshold', linewidth=3)

    # 그래프 제목, x/y 축 라벨 설정 (가독성 있는 폰트 사용)
    ax.set_title(f'Pose Similarity Score Over Time (Avg: {average_similarity:.2f}%)',
                 fontsize=24, fontweight='bold', color='#333333', pad=20)  # 평균 유사도 추가
    ax.set_xlabel('Time (seconds)', fontsize=18, fontweight='bold', color='#333333', labelpad=15)
    ax.set_ylabel('Pose Similarity Score (0-100)', fontsize=18, fontweight='bold', color='#333333', labelpad=15)

    # 눈금 스타일 설정 (더 큰 폰트 및 굵은 색상 사용)
    ax.tick_params(axis='both', which='major', labelsize=14, colors='#333333')

    # 범례 설정 (더 두드러지게 표시)
    ax.legend(loc='upper left', fontsize=14, frameon=True, facecolor='white', edgecolor='black')

    # 그리드 추가 (흰 배경에 대비되는 선명한 그리드)
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')

    # 텍스트 강조 (최대/최소 값)
    min_idx = np.argmin(similarity_list)
    max_idx = np.argmax(similarity_list)
    ax.text(time_values[min_idx], similarity_list[min_idx] - 5, f"Min: {similarity_list[min_idx]:.1f}",
            fontsize=14, color='red', weight='bold', horizontalalignment='center')
    ax.text(time_values[max_idx], similarity_list[max_idx] + 5, f"Max: {similarity_list[max_idx]:.1f}",
            fontsize=14, color='red', weight='bold', horizontalalignment='center')

    # 이미지를 버퍼에 저장
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)

    return HttpResponse(buf.getvalue(), content_type='image/png')

# 웹캠 스트림 처리 함수
def webcam_feed(request):
    reset_events()  # 세션마다 키포인트 초기화 및 이벤트 리셋
    return StreamingHttpResponse(generate_webcam_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# 비디오 스트림 처리 함수
def video_feed(request):
    reset_events()
    response = StreamingHttpResponse(generate_video_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    
    # 응답 헤더에 캐시 비우기 설정 추가
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '-1'
    
    return response

# 메인 페이지 렌더링
def index(request):
    reset_events()  # 세션 초기화
    return render(request, 'index.html', {'audio_file': 'static/audio/kpop1.mp3'})
