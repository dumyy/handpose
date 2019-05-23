import shutil
f=open('end_msra.txt','w+')
for i in range(9):
    for line in open('res_msra_{}.txt'.format(i)):
        f.writelines(line)
f.close()

shutil.copy('end_msra.txt','../../results/')
