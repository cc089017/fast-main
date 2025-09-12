# 예측 결과 시각화 - 김민규 작성

import matplotlib.pyplot as plt
import base64
from io import BytesIO

def build_explain_png(y, feats, features, extras, refs, meta, risk, theta, decision):
    # 파형, 슬라이스, DTW, ZCR/RMS, 판정 카드 등 시각화
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # Top-left: 파형 + 슬라이스
    axs[0, 0].plot(y)
    for s, e in feats.get('edges', []):
        axs[0, 0].axvline(s, color='r', linestyle='--')
    axs[0, 0].set_title("Voiced waveform & slices")

    # Top-right: DTW 5-point (dtw_slice1_mean ~ dtw_slice5_mean)
    dtw_means = [features.get(f'dtw_slice{i}_mean', 0) for i in range(1, 6)]
    axs[0, 1].plot(range(1, 6), dtw_means, marker='o')
    axs[0, 1].set_title("DTW Slice Means")
    axs[0, 1].set_xlabel("Slice")
    axs[0, 1].set_ylabel("DTW Mean")

    # Bottom-left: ZCR/RMS bars
    zcr_rms_labels = ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std']
    zcr_rms_values = [features.get(label, 0) for label in zcr_rms_labels]
    axs[1, 0].bar(zcr_rms_labels, zcr_rms_values)
    axs[1, 0].set_title("ZCR/RMS")

    # Bottom-right: 판정 카드
    axs[1, 1].text(0.1, 0.5, f"Risk: {risk:.3f}\nTheta: {theta:.3f}\nDecision: {decision}", fontsize=14)
    axs[1, 1].axis('off')

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64