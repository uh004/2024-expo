// start.js

document.getElementById('nicknameForm').addEventListener('submit', function(event) {
  event.preventDefault();
  
  // 입력 필드에서 닉네임을 가져오기
  const nickname = document.getElementById('nickname').value;

  // 닉네임을 localStorage에 저장
  localStorage.setItem('nickname', nickname);

  // mainUrl 변수는 HTML 템플릿에서 전달됨
  window.location.href = mainUrl;  // 'main' URL 패턴으로 리다이렉트
});