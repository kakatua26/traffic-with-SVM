## Main Class
## Date : 13-June-2020
## Func : Main Class

import numpy as np
import csv
from sklearn import svm

x = []
y = []
z = []

filename = 'sample_data.csv'

##Reading file and saving to variables
with open(filename) as csvfile:
    f_read = csv.reader(csvfile, delimiter = ',')

    for row in f_read:

        #clean Landmark ('' as 0, 'Perumahan' as 0.1)
        if(row[14] == ''):
            landmark = 0
        elif(row[14] == 'Perumahan'):
            landmark = 0.01
    
        #clean days
        if(row[10] == 'Senin'):
            day = 0.01
        elif(row[10] == 'Selasa'):
            day = 0.02

        #clean labels
        if(row[15] == '-'):
            label = 1
        elif(row[15] == 'Menurunkan Penumpang'):
            label = 2
        elif(row[15] == 'Ngetem'):
            label = 3
        elif(row[15] == 'Macet'):
            label = 4
        x.append([float(row[3]),float(row[4]),float(day),float(row[12].split(':')[0]),float(landmark)])
        y.append(float(label));
        #x = np.append(x, [float(j) for j in row]) -- Comented as only using 1 value

clf = svm.SVC()
clf.fit(x,y)

#extract samples row for testing
for i in (0, 106, 237, 437, 451, 477, 528 ):
    z.append(x[i])

a = clf.predict(z)

print(a)
