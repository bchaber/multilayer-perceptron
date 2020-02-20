# Approximation of the response of two sensors
# raw data from https://github.com/yrdeboer/nn/tree/e4357660225dafb1b307d7149bdc9247947da69a/hagan_case_study_data
ν₁ = [2.6630000e+00   1.9800000e+00   1.4400000e+00   7.1200000e-01   4.5900000e-01   5.4000000e-02   4.0000000e-03   2.0000000e-03   1.0000000e-03   1.0000000e-02   1.3000000e-02   2.5000000e-02   2.5000000e-02   2.8000000e-02   5.6000000e-02   5.5300000e-01   1.7960000e+00   2.8300000e+00   3.1420000e+00   3.8600000e+00   4.2500000e+00   4.6500000e+00   4.9900000e+00   5.3000000e+00   5.4600000e+00   5.6700000e+00   5.7900000e+00   5.8500000e+00   6.0300000e+00   6.2100000e+00   6.3400000e+00   6.4000000e+00   6.4500000e+00   6.5200000e+00   6.5700000e+00   6.6000000e+00   6.6200000e+00   6.6400000e+00   6.6800000e+00   6.6800000e+00   6.6800000e+00   6.6800000e+00   6.6800000e+00   6.6800000e+00   6.6700000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00   6.6600000e+00]
ν₂ = [4.4440000e+00   4.4340000e+00   4.4330000e+00   4.4250000e+00   4.4100000e+00   4.3690000e+00   4.2670000e+00   4.2020000e+00   4.1150000e+00   4.0540000e+00   3.9570000e+00   3.8730000e+00   3.7690000e+00   3.6810000e+00   3.5330000e+00   3.4280000e+00   3.2560000e+00   3.0440000e+00   2.9340000e+00   2.6800000e+00   2.4830000e+00   2.2020000e+00   1.8460000e+00   1.4070000e+00   1.1120000e+00   6.4300000e-01   2.2500000e-01   1.0500000e-01   1.0000000e-01   7.7000000e-02   7.8000000e-02   6.9000000e-02   6.9000000e-02   6.9000000e-02   8.0000000e-02   9.1000000e-02   9.4000000e-02   1.1100000e-01   1.9400000e-01   2.9400000e-01   4.1600000e-01   6.0900000e-01   7.8500000e-01   1.1295000e+00   1.3430000e+00   1.5465000e+00   1.8090000e+00   2.0580000e+00   2.2590000e+00   2.3960000e+00   2.5460000e+00   2.7370000e+00   2.9570000e+00   3.0790000e+00   3.2380000e+00   3.3500000e+00   3.5280000e+00   3.7082000e+00   3.8192000e+00   3.9144000e+00   4.0630000e+00   4.1360000e+00   4.1950000e+00   4.2800000e+00   4.3670000e+00   4.4170000e+00   4.4520000e+00]
y  = [0.0000000e+00   6.4000000e-02   1.1100000e-01   1.7700000e-01   2.0000000e-01   2.6300000e-01   3.2400000e-01   3.5800000e-01   4.0200000e-01   4.3400000e-01   4.8000000e-01   5.2100000e-01   5.7200000e-01   6.1100000e-01   6.6800000e-01   7.0300000e-01   7.5200000e-01   8.0600000e-01   8.2500000e-01   8.7600000e-01   9.1100000e-01   9.5500000e-01   1.0050000e+00   1.0580000e+00   1.0920000e+00   1.1450000e+00   1.1970000e+00   1.2180000e+00   1.2690000e+00   1.3490000e+00   1.4250000e+00   1.4700000e+00   1.5020000e+00   1.5600000e+00   1.6090000e+00   1.6400000e+00   1.6700000e+00   1.6960000e+00   1.7300000e+00   1.7690000e+00   1.7860000e+00   1.8100000e+00   1.8330000e+00   1.8860000e+00   1.9210000e+00   1.9580000e+00   1.9940000e+00   2.0440000e+00   2.0860000e+00   2.1190000e+00   2.1540000e+00   2.2020000e+00   2.2580000e+00   2.2870000e+00   2.3290000e+00   2.3600000e+00   2.4060000e+00   2.4590000e+00   2.4960000e+00   2.5300000e+00   2.5880000e+00   2.6170000e+00   2.6440000e+00   2.6840000e+00   2.7330000e+00   2.7750000e+00   2.8250000e+00]
# normalize
ν₁ ./= 0.5maximum(ν₁)
ν₁ .-= 1.0
ν₂ ./= 0.5maximum(ν₂)
ν₂ .-= 1.0
# prepare data
inputs = hcat(ν₁', ν₂')
targets = y'