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

def replicateScenes(volNames, xmlName):
	xmldoc = minidom.parse(xmlName)
	# for now, just deal with density node
	# in the future we can also combine with scattering albedo density information.
	vs = xmldoc.getElementsByTagName('volume')
	for v in vs:
		if v.getAttribute('name') == 'density':
			# swap out our .vol file
			s = v.getElementsByTagName('string')[0] # this is the node we're interested in
	volNames = [os.path.abspath(volName) for volName in volNames]
	for volName in volNames:
		# possibly 2 matches - either volume:density or volume:albedo
		s.setAttribute('value', volName)
		# extract frame from vol file
		fs = volName.split('_')[-1].split('.vol')[0]
		newXML = xmlName.split('.xml')[0] + '_' + fs + '.xml'
		f = open(newXML,"w")
		xmldoc.writexml(f)
		f.close()

if __name__ == "__main__":
	volNames = sys.argv[1:-1]
	xmlName = sys.argv[-1]
	replicateScenes(volNames, xmlName)
	print('helo world')
