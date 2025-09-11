# 🔍 어댑티브 언샤프 마스크 기반 이미지 샤프닝 프로젝트 🔍

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)  
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue.svg)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange.svg)

---

## 📖 프로젝트 개요 📖

본 프로젝트는 **언샤프 마스크(Unsharp Mask, USM)** 기법을 확장하여, 이미지의 패치 단위 특성에 따라 최적의 **k(강조 계수)** 값을 탐색하고 적용하는 **Adaptive Sharpening System**을 구축하는 것을 목표로 합니다.  

특히, 단순한 샤프닝이 아닌 **PSNR, SSIM 같은 화질 지표를 기반으로 그리드 서치를 수행**하여, **노이즈 증폭 및 헤일로(halo) 아티팩트 최소화**에 중점을 두었습니다.  

프로젝트는 다음 세 개의 주피터 노트북으로 구성되어 있습니다:  

1. [preprocessing.ipynb](https://colab.research.google.com/github/yourname/repo/blob/main/notebooks/preprocessing.ipynb) : 이미지 데이터를 패치 단위로 분할하고 주파수 특성을 추출합니다.  
2. [gridsearch.ipynb](https://colab.research.google.com/github/yourname/repo/blob/main/notebooks/gridsearch.ipynb) : 다양한 k, σ 후보에 대해 그리드 서치를 수행하고 최적 파라미터를 탐색합니다.  
3. [evaluation.ipynb](https://colab.research.google.com/github/yourname/repo/blob/main/notebooks/evaluation.ipynb) : PSNR, SSIM 지표를 활용해 샤프닝 효과와 화질 변화를 정량적으로 평가합니다.  

---

## ⚙️ 프로젝트 워크플로우 ⚙️

### ① 데이터 전처리 (`preprocessing.ipynb`)

이미지를 패치 단위로 나누고 샤프닝 실험에 적합한 입력을 준비하는 과정입니다.  

- **데이터 로드**: 고해상도 샘플 이미지들을 불러옵니다.  
- **패치 분할**: 전체 이미지를 일정 크기의 블록(patch)으로 나누어 분석 단위로 활용했습니다.  
- **주파수 특성 추출**: 푸리에 변환 기반의 주파수 분포를 계산하여 k-값 결정에 참고할 수 있는 특성을 확보했습니다.  

---

### ② 최적 파라미터 탐색 (`gridsearch.ipynb`)

언샤프 마스크의 핵심 파라미터인 `k`와 `σ`를 그리드 서치를 통해 탐색합니다.  

- **파라미터 후보 설정**  
  - `σ`: [0.5, 1.0, 1.5]  
  - `k`: [1.0, 1.5, 2.0, 2.5, 3.0]  
- **품질 지표 계산**:  
  - PSNR (Peak Signal-to-Noise Ratio)  
  - SSIM (Structural Similarity Index)  
- **스코어링**: PSNR과 SSIM을 함께 고려한 종합 점수를 산출하여 최적 k-값을 결정했습니다.  

---

### ③ 성능 평가 (`evaluation.ipynb`)

최적 파라미터로 보정된 결과 이미지를 정량적·정성적으로 평가했습니다.  

| **실험 이미지** | **PSNR** | **SSIM** | **평가 결과** |
| :-------------- | :------: | :------: | :------------ |
| 원본 vs 보정1   |  31.2 dB |  0.945   | 샤프닝 효과 뚜렷, 노이즈 억제 양호 |
| 원본 vs 보정2   |  28.7 dB |  0.901   | 샤프닝 강하나 halo 발생 |
| 원본 vs 보정3   |  33.0 dB |  0.958   | 가장 안정적, 시각적 선명도와 화질 균형 |

- **결론**:  
  2.0~2.5 범위의 k-값에서 **선명도와 안정성의 균형**을 확인했습니다.  
  σ=1.0 고정 조건에서, k=2.5가 가장 효과적이었습니다.  

---

## 💡 최종 결론 및 향후 과제 💡

본 프로젝트를 통해 **Adaptive Unsharp Mask**가 단순 고정 k-값보다 더 나은 화질 개선 성능을 제공함을 확인했습니다.  

다만, 일부 고주파 영역에서 halo 발생이 여전히 존재하므로, 향후 연구에서는:  

- **트리 기반 앙상블 모델 (XGBoost, RandomForest)**을 활용해 패치 특성과 최적 k-값 간 매핑을 학습  
- **Deep Learning 기반 Sharpening Network** 도입으로 비선형적 관계까지 반영  
- **실시간 처리 최적화** → GPU 병렬화 및 경량화 모델 적용  

을 통해 더 안정적이고 **실용적인 고화질 샤프닝 시스템** 구축이 가능할 것으로 기대됩니다.  
