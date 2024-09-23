document.getElementById('nicknameForm').addEventListener('submit', function(event) {
  event.preventDefault(); // 폼 제출 기본 동작 막기

  const nickname = document.getElementById('nickname').value;

  if (nickname.trim() === "") {
      alert("닉네임을 입력해주세요.");
      return;
  }

  // 닉네임을 로컬 스토리지에 저장
  localStorage.setItem('nickname', nickname);

  // 페이지 이동 (다음 페이지로)
  window.location.href = "startnext.html"; // 다음 페이지 URL
});
