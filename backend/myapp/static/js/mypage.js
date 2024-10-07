document.addEventListener('DOMContentLoaded', () => {
  // localStorage에서 저장된 닉네임 불러오기
  const savedNickname = localStorage.getItem('nickname');

  // 기본 예시 데이터 (localStorage에 닉네임이 없을 경우만 사용)
  const playerData = {
      playerName : savedNickname ? savedNickname : "하츄핑", // 저장된 닉네임이 있으면 사용, 없으면 기본값 사용
      playerLevel : 1,
      playTime : "0분",
      highScore : "0점",
      ranking : "상위 ??%",
      favoriteSongs: [
          "아픈건 딱 질색이니까",
          "모기송",
          "Supersonic"
      ]
  }

  // 데이터를 페이지에 적용
  document.getElementById('playerName').textContent = playerData.playerName;
  document.getElementById('playerLevel').textContent = playerData.playerLevel;
  document.getElementById('playTime').textContent = playerData.playTime;
  document.getElementById('highScore').textContent = playerData.highScore;
  document.getElementById('ranking').textContent = playerData.ranking;

  // 즐겨찾는 곡 목록 표시
  const songList = document.getElementById('favoriteSongs');
  songList.innerHTML = '';  // 기존 내용을 지우고 새로 추가
  playerData.favoriteSongs.forEach(song => {
      const li = document.createElement('li');
      li.textContent = song;
      songList.appendChild(li);
  });

  // 로그아웃 버튼에 이벤트 리스너 추가
  document.querySelector('.logout-button').addEventListener('click', () => {
    logout();
  });
});

// 로그아웃 기능
function logout() {
  if (confirm('정말 로그아웃 하시겠습니까?')) {
      localStorage.removeItem('nickname'); // 닉네임 제거
      window.location.href = 'http://127.0.0.1:8000/'; // 지정한 URL로 직접 리다이렉트
  }
}
