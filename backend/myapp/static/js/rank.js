// 카테고리별 랭킹 데이터
const rankings = {
  kpop: [
    { rank: 1, username: '소원', score: 9800 },
    { rank: 2, username: 'Player456', score: 9500 },
    { rank: 3, username: 'Champion789', score: 9200 }
  ],
  challenge: [
    { rank: 1, username: 'Dancer123', score: 10000 },
    { rank: 2, username: 'RhythmMaster', score: 9900 },
    { rank: 3, username: 'BeatKing', score: 9700 }
  ],
  all: [
    { rank: 1, username: 'AllStar1', score: 11000 },
    { rank: 2, username: 'PlayerOne', score: 10500 },
    { rank: 3, username: 'UltimateChamp', score: 10200 }
  ]
};

// 랭킹을 보여주는 함수
function showRanking(category) {
  const tbody = document.getElementById('ranking-body');
  tbody.innerHTML = ''; // 기존 내용을 지우고

  // 선택한 카테고리의 데이터를 사용하여 테이블을 업데이트
  rankings[category].forEach(player => {
    const row = `<tr>
      <td>${player.rank}</td>
      <td>${player.username}</td>
      <td>${player.score}</td>
    </tr>`;
    tbody.innerHTML += row;
  });
}

// 초기 로딩 시 기본 Kpop Shots 랭킹 보여주기
document.addEventListener('DOMContentLoaded', function() {
  showRanking('kpop');
});