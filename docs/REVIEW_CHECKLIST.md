# 서버 실행 전 검토할 결정

코드의 초기 선택값이며, 사용자가 이미 확정한 실험 조건이라는 의미는 아닙니다.

| 항목 | 현재 구현 | 검토 이유 |
|---|---|---|
| ② EMG 표현 | causal filtered EMG의 trailing 50ms RMS 4개 | 원신호 단일 샘플 concat과 실험 정의가 다름 |
| ③ 윈도우 | 과거 200ms, 2000Hz, 400×4 | 지연과 정보량 사이 선택 |
| ③ 인코더 | Conv1d 2층 + pool + Linear, 토큰 1개 | 다중 시간 토큰 방식은 현재 미구현 |
| ③ 토큰 위치 | ACT environment-state 경로 | state·영상 토큰과 같은 encoder attention |
| EMG의 VAE 입력 | ②는 포함, ③은 미포함 | upstream 구조 유지에 따른 차이 |
| 모델 공통 데이터 선별 | EMG 전체 윈도우 유효 프레임 교집합 | baseline에도 같은 프레임 수 적용 |
| 학습 FPS/이미지 해상도 | 30Hz / RGB 320×180 | 원본 30Hz·1280×720의 비율을 유지해 축소 |
| 시계/필터/유효 구간 | 실제 배열 감사 후 causal 수신 기준 적용 | MCU mapping은 여전히 추정치; 생리학적 유효성은 별도 검증 |
| validation | 고정 에피소드, 앞쪽 최대 20 batch의 zero-latent L1 | 전체 validation/성공률 평가는 별도 |
| action 예측/실행 길이 | chunk_size=30, n_action_steps=15 | 세 모델 모두 동일하게 적용; 15스텝 실행 후 재예측 |
| checkpoint | 500 step마다, 최근 3개 | optimizer 포함, 실험당 용량 제한 |
| 배포 형식 | 프로젝트 checkpoint + CycloACTAdapter | 기존 Cyclo loader 자동 호환은 아님 |

로컬 검증은 Python 3.11.5 / torch 2.5.1+cpu에서 진행했습니다.
EMG 정렬/인코더/체크포인트 테스트와 설정 출력은 로컬에서 검사할 수 있습니다.
서버 Python 3.12에서 전체 9개 테스트와 세 모델의 실제 데이터 20 step 학습/재개/복원 검증을 완료했습니다.
검증 결과와 남은 제한은 `reports/VALIDATION.md`를 참고하세요.
