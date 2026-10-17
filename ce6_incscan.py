# ce6_incscan.py —— 缺口随倾角的开合 (win 串行, 运行时改 INC_TGT 合法)
import diyue6 as d
for inc in (23.3, 23.45, 23.58, 23.7, 23.85):
    d.INC_TGT = inc
    print("#" * 100); print("INC_TGT = %.2f°" % inc)
    d.run_win("2024-05-03", 500.0, 110.0, 114.5, 0.1)
