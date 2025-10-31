import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1) 데이터 불러오기 (1열: 에너지 x, 2열: 카운트 y)
url = 'https://raw.githubusercontent.com/nicesw77/com_phys2024/main/hist2.csv'
data = np.loadtxt(url, delimiter=',', skiprows=1)

x = data[:, 0]
y = data[:, 1]

# 2) 가우시안 2개 모델
def double_gaussian(x, a1, mu1, s1, a2, mu2, s2):
    g1 = a1 * np.exp(-((x - mu1)**2) / (2 * s1**2))
    g2 = a2 * np.exp(-((x - mu2)**2) / (2 * s2**2))
    return g1 + g2

# 3) 초기값(그래프 모양 보고 대략)
p0 = [900, 1.0, 0.20, 600, 2.0, 0.35]

# 4) 피팅
params, cov = curve_fit(double_gaussian, x, y, p0=p0, maxfev=100000)
a1, mu1, s1, a2, mu2, s2 = params

# 5) 성분 분해 및 합
g1 = a1 * np.exp(-((x - mu1)**2) / (2 * s1**2))
g2 = a2 * np.exp(-((x - mu2)**2) / (2 * s2**2))
y_fit = g1 + g2

# 6) 면적(적분값)과 생성비
area1 = np.trapz(g1, x)
area2 = np.trapz(g2, x)
ratio = area1 / area2
print(f"[A] a={a1:.3f}, mu={mu1:.4f}, sigma={s1:.4f}")
print(f"[B] a={a2:.3f}, mu={mu2:.4f}, sigma={s2:.4f}")
print(f"Area A={area1:.3f}, Area B={area2:.3f}")
print(f"Yield ratio A:B = {ratio:.3f} : 1")

# 7) 그래프
dx = np.median(np.diff(x))
plt.bar(x, y, width=dx, color='gray', alpha=0.5, label='data')
plt.plot(x, g1, 'r-', label='particle A')
plt.plot(x, g2, 'b-', label='particle B')
plt.plot(x, y_fit, 'k--', label='total fit')
plt.xlabel("Energy"); plt.ylabel("Count"); plt.legend()
plt.title("Gaussian fitting of energy spectrum")
plt.tight_layout()
plt.show()

plt.savefig("fit_result.png", dpi=200)
