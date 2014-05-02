import array

f = open("/gpfs/main/home/evjang/course/cs224/group_final/snow/project/data/scenes/THE_D_0001.vol","r+b")
#f = open("/data/people/evjang/offline_renders/mts_scene/test_0000.vol","r+b")

f.seek(24)
bbox_old = array.array('f')
bbox_old.fromfile(f,6)
print('old bbox : ')
print(bbox_old)
bbox_new = array.array('f', [-0.5, 0.0, -0.5, 0.5, 1.0, 0.5])
#bbox_new = array.array('f',[-0.0300, 0.0030, 0.0294,0.0700, 0.0530, 0.0794 ])


# print('new bbox : ')
# print(bbox_new)
# print((bbox_new[3]+bbox_new[0])/2)
# print((bbox_new[4]+bbox_new[1])/2)
# print((bbox_new[5]+bbox_new[2])/2)
# print('---')
# print(bbox_new[3]-bbox_new[0])
# print(bbox_new[4]-bbox_new[1])
# print(bbox_new[5]-bbox_new[2])
f.seek(24)
bbox_new.tofile(f)
print('done')
