from django.db import models


class Video(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    video = models.FileField(upload_to='videos/', null=True, blank=True)  # FileField 사용
    created_at = models.DateTimeField(auto_now_add=True)
    


class Player(models.Model):
    GENDER_CHOICES = [
        ('M', '남자'),
        ('F', '여자'),
    ]
    
    nickname = models.CharField(max_length=100, primary_key=True)  # 닉네임을 primary_key로 설정
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='M')  # 성별 필드 추가, 기본값 M


    def __str__(self):
        return f"닉네임: {self.nickname}, 성별: {self.get_gender_display()}"
class Post(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    video = models.FileField(upload_to='videos/', null=True, blank=True)  # FileField 사용
    created_at = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='posts')  # 작성자를 Player 모델의 nickname으로 설정


# 댓글 모델 (Comment)
class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')  
    # 댓글이 달린 게시글: Post 모델과 관계 설정. 게시글이 삭제되면 관련 댓글도 삭제됨.
    author = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='comments')  
    # 댓글 작성자: Player 모델과 관계 설정.
    content = models.TextField()  # 댓글 내용
    created_at = models.DateTimeField(auto_now_add=True)  # 댓글 작성 일시 자동 기록

    def __str__(self):
        return f"댓글 작성자: {self.author.nickname}, 내용: {self.content[:20]}"