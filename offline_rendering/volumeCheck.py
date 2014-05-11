# validates volume file

import array

f = open("/data/people/evjang/offline_renders/THE_rerender/THE_D_0000.vol","rb")
#f = open("/data/people/evjang/offline_renders/mts_scene/test_0000.vol","rb")

f.seek(8)
data = array.array('i')
data.fromfile(f,1)
print('xres : ', data)
xres = data[0]
f.seek(12)
data = array.array('i')
data.fromfile(f,1)
print('yres : ', data)
yres = data[0]
f.seek(16)
data = array.array('i')
data.fromfile(f,1)
print('zres : ', data)
zres = data[0]

f.seek(20)
data = array.array('i')
data.fromfile(f,1)
print('num channels : ', data)


f.seek(24)
bbox_old = array.array('f')
bbox_old.fromfile(f,6)
print('bbox : ', bbox_old)

c = 0
for i in range(xres*yres*zres):
	f.seek(48+4*i)
	data = array.array('f')
	data.fromfile(f,1)
	if data[0] > 0.0:
#		print(data[0])
		c += 1

print("%d successes" % c)
