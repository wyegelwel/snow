#! /usr/bin/env python
# Integration plugin for loading snow simulation data into Blender
# and exporting it via Mitsuba.

bl_info = {
    "name": "My Script",
    "description": "Import or Export Snow Simulation",
    "author": "evjang, mliberma, taparson, wyegelwe",
    "version": (1, 0),
    "blender": (2, 69, 0),
    "location": "File > Import > Snow Simulation",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "wyegelwel.github.io/snow",
    "category": "Import-Export"}

import os
import time
import bpy
import sys
from xml.dom import minidom
import mathutils
import math
import array
#import IPython
#IPython.embed()

import pdb


# access to necessary mtsblend classes.
sys.path.append(os.path.dirname('~/.config/blender/2.69/scripts/addons/mtsblend'))
from mtsblend.export.scene import SceneExporterProperties, SceneExporter

class SnowSceneData(object):
	def __init__(self):
		self.msg = 'snow scene says hello'
		
# global data structure
snow_scene_data = SnowSceneData()

# classes for handing import/export of data. Meat of the program is here
class SnowImporter(object):
	"""
	Parses our custom sceneFile format and adds a 
	proxy for the ParticleSystem into the scene.
	"""
	def __init__(self):
		pass

	def parse(self, xmlfile):
		"""
		imports an XML.
		Loads a Cube that represents a proxy for the snow volume.
		Also loads colliders and animates them across their velocity trajectory.
		
		Snow simulator is written with y axis up, but Blender is Z-up,
		so swap the Y axis with the Z axis and we should be good to go.
		"""
		xmldoc = minidom.parse(xmlfile)	
		sdata = SnowSceneData()
		# load SimulationParameters
		spNode = xmldoc.getElementsByTagName('SimulationParameters')[0]
		sdata.timestep = float(spNode.getElementsByTagName('float')[0].getAttribute('value'))

		esNode = xmldoc.getElementsByTagName('ExportSettings')[0]
		sdata.basename = os.path.basename(esNode.getElementsByTagName('string')[0].getAttribute('value'))
		for pNode in esNode.getElementsByTagName('int'):
			val = int(pNode.getAttribute('value'))
			if pNode.getAttribute('name') == 'maxTime':
				sdata.maxTime = val
			elif pNode.getAttribute('name') == 'exportFPS':
				sdata.exportFPS = val
			elif pNode.getAttribute('name') == 'exportDensity':
				sdata.exportDensity = val
			elif pNode.getAttribute('name') == 'exportVelocity':
				sdata.exportVelocity = val

		# add the particle system
		sdata.nframes = math.ceil(sdata.maxTime*sdata.exportFPS)
		bpy.context.scene.frame_end = sdata.nframes
		dirname = os.path.dirname(xmlfile) # assume .vol files are located in same path
		print(dirname + '/' + sdata.basename )
		if sdata.exportDensity:
			sdata.densityVOLs = [(dirname + '/' + sdata.basename + ("_D_%04d.vol" % i) ) for i in range(sdata.nframes)]
			
		if sdata.exportVelocity:
			sdata.velocityVOLs = [(dirname + '/' + sdata.basename + ("_V_%04d.vol" % i) ) for i in range(sdata.nframes)]
		if sdata.exportDensity:
			firstfile = sdata.densityVOLs[0]
		elif sdata.exportVelocity:
			firstfile = sdata.velocityVOLs[0]
		# create particlesystem box
		# infer from volume data
		# f = open(firstfile,"rb")
		# f.seek(24) # beginning of bbox data
		# bbox = array.array('f')
		# bbox.fromfile(f,6)
		# [minX, minY, minZ, maxX, maxY, maxZ] = [i for i in bbox]
		# print(bbox)
		gNode = xmldoc.getElementsByTagName('Grid')[0]
		dNode = gNode.getElementsByTagName('dim')[0]
		dimX = int(dNode.getAttribute('x'))
		dimY = int(dNode.getAttribute('y'))
		dimZ = int(dNode.getAttribute('z'))
		[dimY,dimZ] = [dimZ, dimY]

		pNode = gNode.getElementsByTagName('vector')[0]
		posX = float(pNode.getAttribute('x'))
		posY = float(pNode.getAttribute('y'))
		posZ = float(pNode.getAttribute('z'))
		# swap
		[posY,posZ] = [posZ, posY]

		h = float(gNode.getElementsByTagName('float')[0].getAttribute('value'))	

		# extract position of bounding box from grid position.
		bbox_min = mathutils.Vector([posX, posY, posZ])
		bbox_max = mathutils.Vector([posX+h*dimX, posY+h*dimY, posZ+h*dimZ])
		
		print('original bbox:')
		print(bbox_min)
		print(bbox_max)
		#pdb.set_trace()
		
		bpy.ops.mesh.primitive_cube_add(location=((bbox_min+bbox_max)/2)) 
		bpy.context.object.name = '__snowVolume__'
		container = bpy.data.objects['__snowVolume__']
		container.scale = (bbox_max-bbox_min)/2

		# add the colliders	
		colliderNodes = xmldoc.getElementsByTagName('Collider')
		for cNode in colliderNodes:
			vNodes = cNode.getElementsByTagName('vector')
			cData = {} # collider data
			cType = int(cNode.getAttribute('type'))
			for vNode in vNodes: # parse the collider data
				vector = mathutils.Vector([float(vNode.getAttribute(i)) for i in ['x','y','z']])
				[vector[1],vector[2]] = [vector[2],vector[1]] # swap Y,Z
				cData[vNode.getAttribute('name')] = vector
				
			if cType == 0: # HALF-PLANE
				print('adding half plane')
				up = mathutils.Vector([0,0,1])
				normal = cData['param']
				RotationAxis = up.cross(normal)
				angle = math.acos(normal.dot(up))
				bpy.ops.mesh.primitive_plane_add(location=cData['center'])
				bpy.ops.transform.rotate(value=angle, axis=RotationAxis)
			elif cType == 1: # COLLIDER SPHERE
				print('adding sphere')
				bpy.ops.mesh.primitive_uv_sphere_add(location=cData['center'], size=cData['param'][0])
			
			# print(cData)
			obj = bpy.context.object
			# set the start keyframe
			obj.keyframe_insert(data_path='location',frame=0)
			# set the end keyframe
			obj.location += cData['velocity']*sdata.maxTime
			obj.keyframe_insert(data_path='location',frame=sdata.nframes)

		return sdata
		
class MitsubaSnowExporter(object):
	"""
	Simply makes a call to the mtsblend plugin to export 1 .xml file for each frame 
	of animation. Paths to VOL files are injected during this step.
	"""
	def __init__(self):
		pass

	def export(self, sdata, vol_type, filepath):
		print('export called')
		self.config_blender()
		# As of 30 April 2014, mtsblend's "bpy.ops.export.mitsuba" operator is broken.
		# however rendering obviously works by writing out the correct XMLs, so we will
		# mimic that part of the pipeline.
		scene = bpy.context.scene
		basename = os.path.splitext(os.path.basename(filepath))[0]
		dirname = os.path.dirname(filepath)
		sdata.output_filenames = [(basename + ("_%s_%04d" % (vol_type,i)) ) for i in range(sdata.nframes)]
		print(sdata.output_filenames[0:3])
		# export each frame sequentially. 
		c = bpy.data.objects['__snowVolume__']
		print('location: ', c.location)
		print('dimensions : ', c.dimensions)
		minX = c.location.x - c.dimensions[0]/2
		minY = c.location.y - c.dimensions[1]/2
		minZ = c.location.z - c.dimensions[2]/2
		maxX = c.location.x + c.dimensions[0]/2
		maxY = c.location.y + c.dimensions[1]/2
		maxZ = c.location.z + c.dimensions[2]/2
		"""
		instead of re-writing the bytes in each VOL file, we just set a transformation attribute on 
		the medium interior

		- rotate x -90 degrees
		- translate to desired location
		- scale up 		

		"""
		bbox = array.array('f',[minX, minY, minZ, maxX ,maxY ,maxZ])
		print('new bbox:')
		print(bbox)
		#pdb.set_trace()
		for frame in range(sdata.nframes):
			bpy.context.scene.frame_current=frame
			scene_exporter = SceneExporter()
			scene_exporter.properties.directory = dirname # has to be a full path
			scene_exporter.properties.filename = sdata.output_filenames[frame] # dont add the xml extension, it gets added later.
			scene_exporter.properties.api_type = 'FILE'			# Set export target
			scene_exporter.properties.write_files = True		# Use file write decision from above
			scene_exporter.properties.write_all_files = False		# Use UI file write settings
			scene_exporter.set_scene(scene)
			export_result = scene_exporter.export()

	def config_blender(self):
		# disable logo
		bpy.data.cameras["Camera"].mitsuba_camera.mitsuba_film.banner = False
		# high quality settings
		bpy.data.cameras["Camera"].mitsuba_camera.mitsuba_film.highQualityEdges = False # change this 
		bpy.context.scene.mitsuba_integrator.type = 'volpath_simple'
		bpy.data.cameras["Camera"].mitsuba_camera.mitsuba_film.fileFormat = 'png'
		# HDTV 1080p - lower resolution for prototyping for the time being
		bpy.context.scene.render.resolution_x = 1920 * .25
		bpy.context.scene.render.resolution_y = 1080 * .25
		bpy.context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'

################################################################################################

# Blender UI Operators

class SnowImportOperator(bpy.types.Operator):
	""" Load a Snow Simulation Scene """	# tooltip
	bl_idname = "snow.import"				# unique identifier for buttons and menu items to reference
	bl_label = "Import Snow Simulation"		# display name in interface
	bl_options = {'REGISTER', 'UNDO'}		# enable undo
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'

	filepath = bpy.props.StringProperty(subtype='FILE_PATH', options={'HIDDEN',})

	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		global snow_scene_data
		scene = bpy.context.scene
		sImporter = SnowImporter()
		snow_scene_data = sImporter.parse(self.filepath)
		context.scene.update()
		return {'FINISHED'}

class SnowExportOperator(bpy.types.Operator):
	""" Export Mitsuba Snow """ 
	bl_idname = "snow.export"				
	bl_label = "Export Mitsuba Snow"	
	bl_options = {'REGISTER', 'UNDO'}		
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'

	filepath = bpy.props.StringProperty(subtype='FILE_PATH', options={'HIDDEN',})

	# context groups
	vol_type = bpy.props.EnumProperty(
            name="Volume Data",
            items=(('D', "Density", ""),
                   ('V', "Velocity", "")
                   ),
            default='D',
            )

	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		print('Exporting Snow Scene...')
		mtsExporter = MitsubaSnowExporter()
		mtsExporter.config_blender()
		global snow_scene_data
		mtsExporter.export(snow_scene_data, self.vol_type, self.filepath)
		print('export finished!')
		context.scene.update()
		return {'FINISHED'}

# Blender Menu function bindings to operators

def menu_import_func(self, context):
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator(SnowImportOperator.bl_idname, text=SnowImportOperator.bl_label)

def menu_export_func(self, context):
	self.layout.operator_context = 'INVOKE_DEFAULT'
	self.layout.operator(SnowExportOperator.bl_idname, text=SnowExportOperator.bl_label)

def register():
	bpy.utils.register_class(SnowImportOperator)
	bpy.types.INFO_MT_file_import.append(menu_import_func)
	bpy.utils.register_class(SnowExportOperator)
	bpy.types.INFO_MT_file_export.append(menu_export_func)
	print('Plugin Registered')
	
def unregister():
	bpy.utils.unregister_class(SnowImportOperator)
	bpy.utils.unregister_class(SnowExportOperator)
	print('Plugin Unregistered')

################################################################################################

if __name__ == "__main__":
	print('hello world')
	register()

