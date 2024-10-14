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

