document.getElementById('nicknameForm').addEventListener('submit', function(event) {
  event.preventDefault();
  
  // 입력 필드에서 닉네임을 가져오기
  const nickname = document.getElementById('nickname').value;

  // 닉네임을 localStorage에 저장
  localStorage.setItem('nickname', nickname);

  // 메인 페이지로 리다이렉트
  window.location.href = 'main.html'; // 경로는 필요에 따라 조정하세요.
});