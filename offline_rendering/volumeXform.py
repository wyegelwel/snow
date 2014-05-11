import array

f = open("/data/people/evjang/offline_renders/THE_rerender/THE_D_0000.vol","r+b")
#f = open("/data/people/evjang/offline_renders/mts_scene/test_0000.vol","r+b")

f.seek(24)
bbox_old = array.array('f')
bbox_old.fromfile(f,6)
print('old bbox : ')
print(bbox_old)
bbox_new = array.array('f', [-0.0300, 0.0294, 0.0030, 0.0700, 0.0794, 0.0530])


f.seek(24)
bbox_new.tofile(f)
print('done')
