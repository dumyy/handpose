import shutil
import os

f1=open('test_seq_1.txt','r')
f2=open('test_seq_2.txt','r')
f3=open('lables.txt','a+')

dir_name='2014'

if not os.path.exists('{}/Depth/{}'.format(os.getcwd(),dir_name)):
	os.mkdir('{}/Depth/{}'.format(os.getcwd(),dir_name))

seq1=f1.readlines()
seq2=f2.readlines()


cmb=seq1+seq2

for i in range(len(cmb)):
	str_=cmb[i].split(' ')
	old_name=str_[0]
	cout=str(i).zfill(4)
	new_name='{}/image_{}.png'.format(dir_name,cout)
	str_[0]=new_name
	f3.write(' '.join(str_))
	shutil.copy('{}/Depth/{}'.format(os.getcwd(),old_name),'{}/Depth/{}'.format(os.getcwd(),new_name))

f3.close()
f2.close()
f1.close()







