from django.shortcuts import render, redirect
from django.http import HttpResponse

from django.http import StreamingHttpResponse, FileResponse, Http404
from django.conf import settings
import os
from django.templatetags.static import static

# Create your views here.

def start(request):
  return render(request, 'start.html')

def main(request):
  return render(request, 'main.html')

def gamerule(request):
  return render(request, 'gamerule.html')

def rank(request):
  return render(request, 'rank.html')

def comunity(request):
  return render(request, 'comunity.html')

def write(request):
  return render(request, 'write.html')

from django.shortcuts import render, redirect
from django.http import HttpResponse

def submit_post(request):
    if request.method == "POST":
        title = request.POST.get('title')
        content = request.POST.get('content')
        # 여기에 동영상 및 사진 처리 로직을 추가할 수 있습니다.

        # 처리 후 페이지 리다이렉션
        return redirect('comunity')  # 커뮤니티 페이지로 리다이렉트

    return HttpResponse("Invalid request method.")


def mypage(request):
  return render(request, 'mypage.html')

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

from django.http import StreamingHttpResponse
import cv2
import os

# 웹캠 스트림 처리
def generate_webcam_frames():
    cap = cv2.VideoCapture(0)  # 웹캠 연결

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 프레임을 스트리밍으로 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def webcam_feed(request):
    return StreamingHttpResponse(generate_webcam_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# 비디오 파일 스트림 처리
def generate_video_frames():
    video_path = os.path.join('media', 'video', 'kpop1.mp4')  # 동영상 파일 경로
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 프레임을 스트리밍으로 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(generate_video_frames(), content_type='multipart/x-mixed-replace; boundary=frame')