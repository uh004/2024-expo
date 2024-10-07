from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.start, name='start'),                   # 첫 페이지
    path('main/', views.main, name='main'),                # 메인 페이지
    path('gamerule/', views.gamerule, name='gamerule'),    # 게임방법
    path('rank/', views.rank, name='rank'),                # 랭킹 페이지
    path('comunity/', views.comunity, name='comunity'),    # 커뮤니티 페이지
    path('write/', views.write, name='write'),             # 글 작성 페이지
    path('submit_post/', views.submit_post, name='submit_post'),  # 새 글 등록 경로
    path('mypage/', views.mypage, name='mypage'),          # 마이페이지
    path('choice/', views.choice, name='choice'),          # 선택 페이지
    path('kpop/', views.kpop, name='kpop'),                # K-pop 페이지
    path('kpopcard1/', views.kpopcard1, name='kpopcard1'), # K-pop 카드 1
    path('kpopcard2/', views.kpopcard2, name='kpopcard2'), # K-pop 카드 2
    path('kpopcard3/', views.kpopcard3, name='kpopcard3'), # K-pop 카드 3
    path('shots/', views.shots, name='shots'),             # Shots 페이지
    path('shotscard1/', views.shotscard1, name='shotscard1'), # Shots 카드 1
    path('shotscard2/', views.shotscard2, name='shotscard2'), # Shots 카드 2
    path('shotscard3/', views.shotscard3, name='shotscard3'), # Shots 카드 3
    path('challenge/', views.challenge, name='challenge'), # 챌린지 페이지
    path('challengecard1/', views.challengecard1, name='challengecard1'), # 챌린지 카드 1
    path('challengecard2/', views.challengecard2, name='challengecard2'), # 챌린지 카드 2
    path('challengecard3/', views.challengecard3, name='challengecard3'), # 챌린지 카드 3
    
    path('kpopcard1_start/', views.kpopcard1_start, name='kpopcard1_start'),
]
