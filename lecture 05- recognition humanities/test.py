import numpy as np
import matplotlib.pyplot as plt

tinydict = {'Google': 'www.google.com', 'Runoob': 'www.runoob.com', 'taobao': 'www.taobao.com'}
for r_a in tinydict.items():
    for r_b in tinydict.items():
        print(r_a[1], r_b[1])
