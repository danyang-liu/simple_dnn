# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

data = [578870, 163638, 58185, 22481, 9544, 7688, 248]
labels = ['1', '2', '3', '4', '5', '6-10', '10+']
#percent = ['68.9%','19.5%','6.9%','2.7%','1.1%','0.9%','0.03%']
percent = [68.9,19.5,6.9,2.7,1.1,0.9,0.03]

plt.bar(range(len(data)), data, tick_label=labels)

for a,b in zip(range(len(data)),data):
    plt.text(a, b,'%.2f' % (float(b)/8401)+'%' ,ha='center',va='bottom',fontsize=11)
plt.xlabel("Click num(unique doc)")
plt.ylabel("User num")
plt.show()
