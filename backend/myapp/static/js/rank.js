document.addEventListener('DOMContentLoaded', function() {
  // 데이터베이스에서 플레이어 랭킹 데이터를 가져옴
  fetch('/get_rank_data/')
    .then(response => response.json())
    .then(players => {
      const tbody = document.getElementById('ranking-body');
      tbody.innerHTML = ''; // 기존 내용을 지움

      // Django에서 전달된 이미지 경로 가져오기
      const goldMedal = tbody.getAttribute('data-goldmedal');
      const silverMedal = tbody.getAttribute('data-silvermedal');
      const bronzeMedal = tbody.getAttribute('data-bronzemedal');

      console.log('Gold Medal:', goldMedal);  // 경로 출력
      console.log('Silver Medal:', silverMedal);
      console.log('Bronze Medal:', bronzeMedal);

      // 가져온 플레이어 데이터를 사용하여 테이블을 동적으로 생성
      players.forEach((player, index) => {
        let crownImg = '';

        // 순위에 따라 메달 이미지를 선택
        if (index === 0) {
          crownImg = `<img src="${goldMedal}" alt="금메달" class="medal-icon">`; // 금메달
        } else if (index === 1) {
          crownImg = `<img src="${silverMedal}" alt="은메달" class="medal-icon">`; // 은메달
        } else if (index === 2) {
          crownImg = `<img src="${bronzeMedal}" alt="동메달" class="medal-icon">`; // 동메달
        }

        // 각 플레이어의 정보를 테이블에 추가
        const row = `
          <tr>
            <td>
              <div class="rank-container">
                ${crownImg}
                <span class="rank-number">${index + 1}</span>
              </div>
            </td>
            <td>${player.nickname}</td>
            <td class="score">${player.score.toFixed(2)}</td>
          </tr>`;
        tbody.innerHTML += row;
      });
    })
    .catch(error => {
      console.error('Error fetching rank data:', error);
    });
});
