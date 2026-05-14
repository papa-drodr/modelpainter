# ModelPainter
---

## 프로젝트 소개

프라모델을 조립하고 나서 "이 색으로 칠하면 어떨까?" 라는 생각에서 시작했습니다.
GB4라는 게임보고 프로젝트로 하면 좋겠다라고 생각했습니다.

ModelPainter는 영상 하나만 있으면:
1. 자동으로 프레임을 추출하고
2. NeRF(Neural Radiance Fields)로 3D 모델을 생성한 뒤
3. 메시 단위로 원하는 색상을 입혀볼 수 있는 프로그램입니다.

---

## 파이프라인

```
영상 입력 (.mp4)
    ↓
1. Frame Extraction     — 영상 길이에 따라 150장 균일 샘플링
    ↓
2. pycolmap             — 카메라 intrinsic + pose 자동 추정
    ↓
3. NeRF 학습            — 직접 구현 (PyTorch)
    ↓
4. Mesh 추출            — Marching Cubes로 3D 메시 생성
    ↓
5. 색상 편집            — 메시 단위 RGB / HSV 색상 수정
```

---

## 목표

- NeRF를 직접 구현하여 3D 재건 원리 이해
- 메시 단위에서 RGB 및 HSV를 이용한 색상 편집 구현
- 영상 입력부터 색상 편집까지 end-to-end 파이프라인 완성