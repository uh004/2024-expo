from django.db import models

class Video(models.Model):
    title = models.CharField(max_length=100)
    video_url = models.URLField(max_length=200)  # 비디오 URL 저장 필드
    uploaded_at = models.DateTimeField(auto_now_add=True)
