#! /usr/bin/env python
# simple export script that takes an .xml mitsuba scene and duplicates it. 
# a different .vol is inserted into each copy for animation playback
# the next step is to use this script on the .xml files that Mitsuba's Blender integration plugin exports

# simple hack, just goes in and replaces 

# usage : $ python insertVolumes.py smoke_*.vol base_scene.xml

import sys
#import pdb as pdb
import os
from xml.dom import minidom

# class SnowSceneData(object):
# 	def __init__(self):
# 		self.msg = 'snow scene says hello'

# def parse(self, xmlfile):
# 		xmldoc = minidom.parse(xmlfile)	
# 		sdata = SnowSceneData()
# 		# load SimulationParameters
# 		spNode = xmldoc.getElementsByTagName('SimulationParameters')[0]
# 		sdata.timestep = float(spNode.getElementsByTagName('float')[0].getAttribute('value'))

# 		esNode = xmldoc.getElementsByTagName('ExportSettings')[0]
# 		sdata.basename = os.path.basename(esNode.getElementsByTagName('string')[0].getAttribute('value'))
# 		for pNode in esNode.getElementsByTagName('int'):
# 			val = int(pNode.getAttribute('value'))
# 			if pNode.getAttribute('name') == 'maxTime':
# 				sdata.maxTime = val
# 			elif pNode.getAttribute('name') == 'exportFPS':
# 				sdata.exportFPS = val
# 			elif pNode.getAttribute('name') == 'exportDensity':
# 				sdata.exportDensity = val
# 			elif pNode.getAttribute('name') == 'exportVelocity':
# 				sdata.exportVelocity = val

# 		# add the particle system
# 		sdata.nframes = math.ceil(sdata.maxTime*sdata.exportFPS)
# 		bpy.context.scene.frame_end = sdata.nframes
# 		dirname = os.path.dirname(xmlfile)
# 		print(dirname + '/' + sdata.basename )
# 		if sdata.exportDensity:
# 			sdata.densityVOLs = [(dirname + '/' + sdata.basename + ("_D_%04d.vol" % i) ) for i in range(sdata.nframes)]
			
# 		if sdata.exportVelocity:
# 			sdata.velocityVOLs = [(dirname + '/' + sdata.basename + ("_V_%04d.vol" % i) ) for i in range(sdata.nframes)]
# 		if sdata.exportDensity:
# 			firstfile = sdata.densityVOLs[0]
# 		elif sdata.exportVelocity:
# 			firstfile = sdata.velocityVOLs[0]
# 		gNode = xmldoc.getElementsByTagName('Grid')[0]
# 		dNode = gNode.getElementsByTagName('dim')[0]
# 		dimX = int(dNode.getAttribute('x'))
# 		dimY = int(dNode.getAttribute('y'))
# 		dimZ = int(dNode.getAttribute('z'))
# 		pNode = gNode.getElementsByTagName('vector')[0]
# 		posX = float(pNode.getAttribute('x'))
# 		posY = float(pNode.getAttribute('y'))
# 		posZ = float(pNode.getAttribute('z'))
# 		h = float(gNode.getElementsByTagName('float')[0].getAttribute('value'))	

# 		bbox_min = mathutils.Vector([posX, posY, posZ])
# 		bbox_max = mathutils.Vector([posX+h*dimX, posY+h*dimY, posZ+h*dimZ])
# 		print('bbox min: ',bbox_min)
# 		print('bbox max: ',bbox_max)
# 		bpy.ops.mesh.primitive_cube_add(location=((bbox_min+bbox_max)/2)) 
# 		bpy.context.object.name = '__snowVolume__'
# 		container = bpy.data.objects['__snowVolume__']
# 		container.scale = (bbox_max-bbox_min)/2

# 		# add the colliders	
# 		colliderNodes = xmldoc.getElementsByTagName('Collider')
# 		for cNode in colliderNodes:
# 			vNodes = cNode.getElementsByTagName('vector')
# 			cData = {} # collider data
# 			cType = int(cNode.getAttribute('type'))
# 			for vNode in vNodes: # parse the collider data
# 				vector = mathutils.Vector([float(vNode.getAttribute(i)) for i in ['x','y','z']])
# 				cData[vNode.getAttribute('name')] = vector
				
# 			if cType == 0: # HALF-PLANE
# 				print('adding half plane')
# 				up = mathutils.Vector([0,1,0])
# 				normal = cData['param']
# 				RotationAxis = up.cross(normal)
# 				angle = math.acos(normal.dot(up))
# 				bpy.ops.mesh.primitive_plane_add(location=cData['center'],rotation=(-math.pi/2,0,0))
# 				bpy.ops.transform.rotate(value=angle, axis=RotationAxis)
# 			elif cType == 1: # COLLIDER SPHERE
# 				print('adding sphere')
# 				bpy.ops.mesh.primitive_uv_sphere_add(location=cData['center'], size=cData['param'][0])
			
# 			print(cData)
# 			obj = bpy.context.object
# 			# set the start keyframe
# 			obj.keyframe_insert(data_path='location',frame=0)
# 			# set the end keyframe
# 			obj.location += cData['velocity']*sdata.maxTime
# 			obj.keyframe_insert(data_path='location',frame=sdata.nframes)


import array

def replicateScenes(volNames, xmlName):
	xmldoc = minidom.parse(xmlName)
	# for now, just deal with density node
	# in the future we can also combine with scattering albedo density information.
	vs = xmldoc.getElementsByTagName('volume')
	for v in vs:
		if v.getAttribute('name') == 'density':
			# swap out our .vol file
			sn = v.getElementsByTagName('string')[0] # this is the node we're interested in
	volNames = [os.path.abspath(volName) for volName in volNames]

	for volName in volNames:
		print(volName)
		f = open(volName,"r+b")
		f.seek(24)
		bbox_old = array.array('f')
		bbox_old.fromfile(f,6)
		# xmin ymin zmin xmax ymax zmax
		w = bbox_old[3]-bbox_old[0]
		h = bbox_old[4]-bbox_old[1]
		d = bbox_old[5]-bbox_old[2]
		s = 1/max([w,h,d])

		newCenter = [0, 0.5, 0]
		bbox_new = array.array( 'f', [newCenter[0]-s*w/2, 0, newCenter[2]-s*d/2, newCenter[0]+s*w/2, s*h, newCenter[2]+s*d/2] );
		f.seek(24)
		bbox_new.tofile(f)
		
		# possibly 2 matches - either volume:density or volume:albedo
		sn.setAttribute('value', volName)
		# extract frame from vol file
		fs = volName.split('_')[-1].split('.vol')[0] # base
		newXML = xmlName.split('.xml')[0] + '_' + fs + '.xml' # new xml file
		f = open(newXML,"w")
		xmldoc.writexml(f)
		f.close()


if __name__ == "__main__":
	volNames = sys.argv[1:-1] # second all the way to the second to last
	xmlName = sys.argv[-1]
	replicateScenes(volNames, xmlName)
	print('done')
