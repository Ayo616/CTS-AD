

tp = 64
fn = 75

fp = 36
tn = 1925

tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

a = (1/2)*tpr*fpr
b =  (1/2)*(tpr+1)*(1-fpr)
print(a+b)