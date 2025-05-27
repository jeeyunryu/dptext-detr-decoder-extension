import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# 좌표 배열
points = np.array([
    [346.  , 547.  ],
    [354.74, 541.31],
    [363.27, 535.38],
    [371.53, 529.15],
    [379.49, 522.55],
    [387.09, 515.54],
    [394.28, 508.04],
    [401.  , 500.  ],
    [422.  , 512.  ],
    [415.59, 521.73],
    [408.37, 531.45],
    [400.32, 540.73],
    [391.46, 549.14],
    [381.79, 556.28],
    [371.3 , 561.7 ],
    [360.  , 565.  ]
])

# points = np.array([[422.  , 512.  ],
#        [415.59, 521.73],
#        [408.37, 531.45],
#        [400.32, 540.73],
#        [391.46, 549.14],
#        [381.79, 556.28],
#        [371.3 , 561.7 ],
#        [360.  , 565.  ],
#        [346.  , 547.  ],
#        [354.74, 541.31],
#        [363.27, 535.38],
#        [371.53, 529.15],
#        [379.49, 522.55],
#        [387.09, 515.54],
#        [394.28, 508.04],
#        [401.  , 500.  ]])

x = points[:, 0]
y = points[:, 1]

# 이미지 좌표계로 맞추기 위해 y축 뒤집기
y_flipped = -y

plt.figure(figsize=(8, 6))

# 모든 점들을 순서대로 연결 (시작-끝 제외 안 함)
plt.plot(x, y_flipped, 'o-', color='blue', label='Path')

# 시작점
plt.scatter(x[0], -y[0], color='green', s=100, label='Start')

# 끝점
plt.scatter(x[-1], -y[-1], color='red', s=100, label='End')

# 시작-끝점 간 직접 연결을 시각적으로 제거하려면 다음과 같이 따로 그리기보단 전체 경로는 유지
# 만약 정말 그 선 하나만 지우고 싶다면 segment들을 하나씩 그려야 합니다 (복잡)

plt.title('2D Path (Image Coordinate System)')
plt.xlabel('X')
plt.ylabel('Y (image down)')
plt.grid(True)
plt.legend()
plt.axis('equal')
# plt.gca().invert_yaxis()  # 이미지 좌표계처럼 y축 아래로 증가

plt.savefig('output.png', dpi=300)
