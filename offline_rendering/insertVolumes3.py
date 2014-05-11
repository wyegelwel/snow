#! /usr/bin/env python
# simple export script that takes an .xml mitsuba scene and duplicates it. 
# a different .vol is inserted into each copy for animation playback

# usage : $ python insertVolumes.py smoke_*.vol base_scene.xml sim_file.xml
# Example:
# python insertVolumes3.py /data/people/evjang/offline_renders/THE/THE_D_*.vol /data/people/evjang/offline_renders/THE/cbox/cbox.xml ~/course/cs224/group_final/snow/project/data/scenes/THE.xml
# replicated mitsuba XMLs are written to same dir as base_scene.xml


import sys
import os
from xml.dom import minidom
import array
import math
import numpy as np
import ipdb as pdb

def get_export_settings(xmldoc):
	sdata = {}
	esNode = xmldoc.getElementsByTagName('ExportSettings')[0]
	sdata['basename'] = os.path.basename(esNode.getElementsByTagName('string')[0].getAttribute('value'))
	for pNode in esNode.getElementsByTagName('int'):
		val = int(pNode.getAttribute('value'))
		if pNode.getAttribute('name') == 'maxTime':
			sdata['maxTime'] = val
		elif pNode.getAttribute('name') == 'exportFPS':
			sdata['exportFPS'] = val
		elif pNode.getAttribute('name') == 'exportDensity':
			sdata['exportDensity'] = val
		elif pNode.getAttribute('name') == 'exportVelocity':
			sdata['exportVelocity'] = val
	return sdata

def get_colliders(xmldoc):
	colliders = []
	colliderNodes = xmldoc.getElementsByTagName('Collider')
	for cNode in colliderNodes:
		vNodes = cNode.getElementsByTagName('vector')
		cData = {} # collider data
		cType = int(cNode.getAttribute('type'))
		for vNode in vNodes: # parse the collider data
			vector = np.array([float(vNode.getAttribute(i)) for i in ['x','y','z']])
			cData[vNode.getAttribute('name')] = vector
		cData['cType'] = cType
		colliders.append(cData)
	return colliders

def get_bbox_from_xml(xmldoc):
	gNode = xmldoc.getElementsByTagName('Grid')[0]
	dNode = gNode.getElementsByTagName('dim')[0]
	dimX = int(dNode.getAttribute('x'))
	dimY = int(dNode.getAttribute('y'))
	dimZ = int(dNode.getAttribute('z'))
	pNode = gNode.getElementsByTagName('vector')[0]
	posX = float(pNode.getAttribute('x'))
	posY = float(pNode.getAttribute('y'))
	posZ = float(pNode.getAttribute('z'))
	h = float(gNode.getElementsByTagName('float')[0].getAttribute('value'))	
	# extract position of bounding box from grid position.
	bbox_old = [posX, posY, posZ, posX+h*dimX, posY+h*dimY, posZ+h*dimZ]
	return bbox_old

def createXformNode(sceneXMLDoc):
	xform = sceneXMLDoc.createElement("transform")
	xform.setAttribute("name","toWorld")
	return xform

def appendXform(sceneXMLDoc,xformNode,xformType,xyz,angle=None):
	node = sceneXMLDoc.createElement(xformType)
	axes= ["x","y","z"]
	for i in range(3):
		node.setAttribute(axes[i],str(xyz[i]))
	if angle is not None:
		node.setAttribute("angle",str(angle))
	xformNode.appendChild(node)


def replicateScenes(volNames, sceneXMLName, xmlName):
	xmldoc = minidom.parse(xmlName)
	sdata = get_export_settings(xmldoc) # get FPS
	

	# cbox.xml path
	sceneXMLDoc = minidom.parse(sceneXMLName)
	vs = sceneXMLDoc.getElementsByTagName('volume')
	# we add materials, colliders, etc. to this node
	sceneNode = sceneXMLDoc.getElementsByTagName('scene')[0]

	# only swap out density attribute, leave albedo and orientation alone for now
	vNode = None
	for v in vs:
		if v.getAttribute('name') == 'density':
			vNode = v

	sn = vNode.getElementsByTagName('string')[0] # this is the node we're interested in

	volNames = [os.path.abspath(volName) for volName in volNames] # paths

	# cMatNode = xmldoc.createElement("")
	# sceneNode.appendChild(cMatNode)
	
	# read bounding box from first VOL data, just in case the grid info we saved in xmlName is inconsistent
	# with what was actually written during simulation.
	f = open(volNames[0],"r+b")
	f.seek(24)
	bbox_old = array.array('f')
	bbox_old.fromfile(f,6)
	
	w = bbox_old[3]-bbox_old[0]
	h = bbox_old[4]-bbox_old[1]
	d = bbox_old[5]-bbox_old[2]
	oldCenter = [(bbox_old[3]+bbox_old[0])/2, (bbox_old[4]+bbox_old[1])/2, (bbox_old[5]+bbox_old[2])/2]
	
	# set 1.0 to fill half the scene, set 2.0 to fill half the cornell box.
	# terrible hack
	s = 2.0/max([w,h,d])
	# instead of writing bbox, just apply transformation to the density node
	# print("s = %f" % (s))
	# bbox_new = array.array( 'f', [-s*w/2, 0, -s*d/2, s*w/2, s*h, s*d/2] );
	
	# create xform node for the volume
	xform = createXformNode(sceneXMLDoc)
	# transform original bbox to origin
	appendXform(sceneXMLDoc, xform, "translate", [-i for i in oldCenter])
	# scale up to fit size of box
	appendXform(sceneXMLDoc, xform, "scale", [s]*3)
	# move it up s*h/2 : note, this corresponds to the Cornell box scene
	appendXform(sceneXMLDoc, xform, "translate", [0,s*h/2,0])
	vNode.appendChild(xform)

	spf = 1.0/float(sdata['exportFPS'])
	xyz= ["x","y","z"]

	########
	######## COLLIDERS 
	########

	cNodes = [] # collider XML nodes reference
	colliders = get_colliders(xmldoc) # collider data
	

	# append colliders
	for cData in colliders:
		c = sceneXMLDoc.createElement("shape")
		# each collider associated with an xform, which as twofold purpose:
		# 1) apply local transformations provided in snow simulation.
		# 2) fit to proper location in cornell box
		x = createXformNode(sceneXMLDoc)
		
		if cData['cType'] == 0: # HALF-PLaANE
			c.setAttribute("type","rectangle")
			upV = np.array([0,1,0])
			normal = cData['param']
			RotationAxis = np.cross(upV,normal)
			if not np.any(RotationAxis):
				# prevent undetermined rotation axis issues
				RotationAxis = np.array([1,0,0])
			angle = math.acos(np.dot(normal,upV)) * 180/math.pi
			appendXform(sceneXMLDoc, x, "rotate", [1,0,0], -90) # planes in mitsuba are by default XY
			appendXform(sceneXMLDoc, x, "scale", 3*[1])# arbitrary for now. see what size plane is reasonable
			appendXform(sceneXMLDoc, x, "rotate", RotationAxis, angle)
			appendXform(sceneXMLDoc, x, "translate", cData["center"])
			mat = sceneXMLDoc.createElement("ref")
			mat.setAttribute("id","purple")

		elif cData['cType'] == 1: # COLLIDER SPHERE
			x = sceneXMLDoc.createElement("transform")
			x.setAttribute("name","toWorld")
			c.setAttribute("type","sphere")
			appendXform(sceneXMLDoc, x, "scale", 3*[cData['param'][0]]) # may require calibration??
			appendXform(sceneXMLDoc, x, "translate", cData["center"])
			mat = sceneXMLDoc.createElement("ref")
			mat.setAttribute("id","gold")

		# step 2) apply same transformation as grid volume so that it is in same place
		# relative to the original volume
		appendXform(sceneXMLDoc, x, "translate", [-i for i in oldCenter])
		appendXform(sceneXMLDoc, x, "scale", [s]*3)
		appendXform(sceneXMLDoc, x, "translate", [0,s*h/2,0])

		c.appendChild(x)
		c.appendChild(mat)

		sceneNode.appendChild(c)
		cNodes.append(c) # maintain easy access to these so we can modify later.

	# write each VOL file
	axes = ["x","y","z"]

	for i, volName in enumerate(volNames):
		# possibly 2 matches - either volume:density or volume:albedo
		sn.setAttribute('value', volName)
		for c_idx in range(len(cNodes)):
			# apply the velocity
			cData = colliders[c_idx]
			cNode = cNodes[c_idx]
			newPos = cData['center'] + spf * i * cData['velocity']
			# the first translate node corresponds to position in original space
			t = cNode.getElementsByTagName("translate")[0] 
			for a in range(3):
				t.setAttribute(axes[a],str(newPos[a]))

		# extract frame from vol file
		fs = volName.split('_')[-1].split('.vol')[0] # base
		newXML = sceneXMLName.split('.xml')[0] + '_' + fs + '.xml' # new xml file
		print(newXML)
		f = open(newXML,"w")
		f.write('\n'.join([line for line in sceneXMLDoc.toprettyxml(indent=' '*2).split('\n') if line.strip()]))
		f.close()

if __name__ == "__main__":
	volNames = sys.argv[1:-2] # second all the way to the second to last
	sceneXMLName = sys.argv[-2] # cbox file
	xmlName = sys.argv[-1] # our custom snow xml
	replicateScenes(volNames, sceneXMLName, xmlName)
	print('done')
