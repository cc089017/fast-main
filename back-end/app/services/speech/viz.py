# 예측 결과 시각화 - 김민규 작성

# 예측 결과를 시각화(PNG)로 생성하는 함수

import matplotlib.pyplot as plt
import io

def build_explain_png(y, feats, features, extras, refs, meta, risk, theta, decision):
    # 파형, 슬라이스, DTW, ZCR/RMS, 판정 카드 등 시각화
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    # Top-left: 파형 + 슬라이스
    axs[0,0].plot(y)
    for s, e in feats['edges']:
        axs[0,0].axvline(s, color='r', linestyle='--')
    axs[0,0].set_title("Voiced waveform & slices")
    # Top-right: DTW 5-point
    axs[0,1].plot(features['dtw_slice1_mean':'dtw_slice5_mean'])
    axs[0,1].set_title("DTW means")
    # Bottom-left: ZCR/RMS bars
    axs[1,0].bar(['zcr_mean','zcr_std','rms_mean','rms_std'],
                 [features['zcr_mean'],features['zcr_std'],features['rms_mean'],features['rms_std']])
    axs[1,0].set_title("ZCR/RMS")
    # Bottom-right: 판정 카드
    axs[1,1].text(0.1, 0.5, f"Risk: {risk:.3f}\nTheta: {theta:.3f}\nDecision: {decision}", fontsize=14)
    axs[1,1].axis('off')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf.read()