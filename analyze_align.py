"""分析对齐效果"""
import cv2, numpy as np

src = cv2.imread('/tmp/dinet_aligned_src.png')
ref = cv2.imread('/tmp/dinet_aligned_ref.png')
out = cv2.imread('/tmp/dinet_rendered_256.png')

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

_, src_th = cv2.threshold(src_gray, 200, 255, cv2.THRESH_BINARY_INV)
_, ref_th = cv2.threshold(ref_gray, 200, 255, cv2.THRESH_BINARY_INV)

sm = cv2.moments(src_th)
rm = cv2.moments(ref_th)

print('=== 新对齐(Dlib参考点) ===')
if sm['m00'] > 0:
    cx = sm['m10'] / sm['m00']
    cy = sm['m01'] / sm['m00']
    print(f'Source质心: ({cx:.0f}, {cy:.0f}) 期望=(128, 130)')
if rm['m00'] > 0:
    cx = rm['m10'] / rm['m00']
    cy = rm['m01'] / rm['m00']
    print(f'Ref质心:   ({cx:.0f}, {cy:.0f}) 期望=(128, 130)')

out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
_, out_th = cv2.threshold(out_gray, 200, 255, cv2.THRESH_BINARY_INV)
om = cv2.moments(out_th)
if om['m00'] > 0:
    cx = om['m10'] / om['m00']
    cy = om['m01'] / om['m00']
    print(f'Output质心:({cx:.0f}, {cy:.0f}) 期望=(128, 130)')

white_pct = (out_gray > 240).mean() * 100
print(f'白色像素(>240): {white_pct:.1f}% (旧: 66%)')

print(f'\n四角: TL={out[0,0]} TR={out[0,-1]} BL={out[-1,0]} BR={out[-1,-1]}')

print(f'额头mask位(20:70,55:201) = mean={out[20:70,55:201].mean():.0f}')
print(f'眼位(80:105,55:201) = mean={out[80:105,55:201].mean():.0f}')
print(f'嘴位(155:200,30:226)= mean={out[155:200,30:226].mean():.0f}')

print('\n逐行(out B通道):')
for y in [0, 45, 90, 130, 186, 230, 255]:
    row = out[y, :, 0]
    print(f'  y={y:3d}: mean={row.mean():.0f} dark%={(row<50).mean()*100:.0f}%')
