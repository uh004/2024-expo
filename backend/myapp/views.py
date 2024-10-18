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

# 웹캠 프레임 생성 및 포즈 감지
def generate_webcam_frames():
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("Error: Cannot access webcam.")
        return

    with mp_pose_webcam.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose1:
        while cap1.isOpened() and not stop_event.is_set():
            ret1, frame1 = cap1.read()
            if not ret1:
                break

            # 프레임을 좌우 반전 및 크기 조정
            frame1 = cv2.flip(frame1, 1)
            frame1 = cv2.resize(frame1, (640, 480))
            frame1.flags.writeable = False
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results1 = pose1.process(frame1_rgb)
            frame1.flags.writeable = True

            if results1.pose_landmarks:
                mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose_webcam.POSE_CONNECTIONS)
                keypoints = extract_keypoints(results1.pose_landmarks.landmark, range(33))
                keypoints_sequence1.append(l2_normalize(keypoints))

            ret, buffer = cv2.imencode('.jpg', frame1)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap1.release()

def get_video_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return 30  # 기본 FPS 값으로 30을 반환 (비디오가 열리지 않으면 기본값 사용)
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS 가져오기
    cap.release()
    return fps

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

# 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# 포즈 비교 함수 (DTW를 사용하여 두 시퀀스의 유사도 계산)
def compare_pose_dtw():
    if len(keypoints_sequence1) == 0 or len(keypoints_sequence2) == 0:
        return []

    similarity_list = []  # 유사도를 저장할 리스트

    for key in range(33):  # 각 관절 인덱스를 처리
        sequence1 = np.array([kp[key] for kp in keypoints_sequence1])  # 첫 번째 비디오의 키포인트 시퀀스
        sequence2 = np.array([kp[key] for kp in keypoints_sequence2])  # 두 번째 비디오의 키포인트 시퀀스

        # DTW를 사용하여 두 시퀀스 정렬
        _, path = fastdtw(sequence1, sequence2, dist=euclidean)
        aligned_sequence1 = [sequence1[idx1] for idx1, _ in path]  # 첫 번째 시퀀스 정렬
        aligned_sequence2 = [sequence2[idx2] for _, idx2 in path]  # 두 번째 시퀀스 정렬

        # 코사인 유사도 계산
        cosine_similarities = [cosine_similarity(p1, p2) for p1, p2 in zip(aligned_sequence1, aligned_sequence2)]
        average_similarity = np.mean(cosine_similarities) * 100  # 평균 코사인 유사도를 100점 만점으로 변환
        similarity_list.append(average_similarity)

    return similarity_list

# 그래프 생성 함수
def graph(request):
    similarity_list = compare_pose_dtw()

    if not similarity_list:
        return HttpResponse("No similarity data available.")

    # fps = 10  # 비디오 FPS (수정 가능)
    fps = get_video_fps(video_file)
    
    # FPS에 기반한 시간 축 생성
    time_values = np.arange(0, len(similarity_list)) * (1 / fps)

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_values, similarity_list, label='Pose Similarity Score after DTW', color='blue', linewidth=2.5, marker='o')
    ax.axhline(y=90, color='red', linestyle='--', label='90 Score Threshold', linewidth=2)
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pose Similarity Score (0-100)', fontsize=14, fontweight='bold')
    ax.set_title('Pose Similarity Score Over Time', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='upper left', fontsize=12)

    # 이미지를 버퍼에 저장
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)

    return HttpResponse(buf.getvalue(), content_type='image/png')

# 메인 페이지 렌더링
def index(request):
    reset_events()  # 세션 초기화
    return render(request, 'index.html', {'audio_file': 'static/audio/kpop1.mp3'})
