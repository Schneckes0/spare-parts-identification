import bpy
import os
import math
import glob
import random
from mathutils import Vector

# Set your paths here
models_directory = r'C:\Users\'
output_directory = r'C:\Users\'


def clear_scene():
    """
    Reset the current Blender scene to factory settings.
    This removes all objects, materials, etc. from the scene.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


def align_camera(camera_obj, target_point):
    """
    Rotate the camera object so that it points towards the target_point.
    :param camera_obj: The camera object to align.
    :param target_point: A mathutils.Vector indicating where the camera should look.
    """
    direction = target_point - camera_obj.location
    rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rotation_quaternion.to_euler()


def set_white_background():
    """
    Set the world background to a plain white color. 
    Uses the World node setup for background color and strength.
    """
    # If there's no World data block, create a new one
    if not bpy.data.worlds:
        bpy.data.worlds.new("World")
    world = bpy.data.worlds.get("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Get or create the Background node
    bg_node = nodes.get("Background")
    if not bg_node:
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.location = (0, 0)

    # Set the background color to white and strength to 5.0
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs['Strength'].default_value = 5.0

    # Get or create the World Output node
    world_output = nodes.get("World Output")
    if not world_output:
        world_output = nodes.new(type='ShaderNodeOutputWorld')
        world_output.location = (300, 0)

    # Clear existing links and link background to the World Output
    links.clear()
    links.new(bg_node.outputs['Background'], world_output.inputs['Surface'])


def setup_freestyle(max_dimension):
    """
    Enable and configure Freestyle to draw black outlines around the object.
    The line thickness scales according to the object's max dimension.
    :param max_dimension: The largest dimension of the model (float).
    """
    # Enable Freestyle
    bpy.context.scene.render.use_freestyle = True

    # Freestyle settings at view layer level
    view_layer = bpy.context.view_layer
    freestyle = view_layer.freestyle_settings
    freestyle.as_render_pass = False  # Directly include lines in the final render

    # Remove all existing line sets
    while freestyle.linesets:
        freestyle.linesets.remove(freestyle.linesets[0])

    # Create a new line set
    line_set = freestyle.linesets.new("LineSet")
    line_set.select_silhouette = True
    line_set.select_border = False
    line_set.select_crease = True
    line_set.select_edge_mark = False

    # Set line style properties
    line_style = line_set.linestyle
    line_style.use_nodes = False
    line_style.color = (0, 0, 0)  # Black lines
    line_style.thickness = max_dimension * 0.01  # Thickness proportional to model size
    line_style.thickness_position = 'CENTER'


def enable_gpu():
    """
    Configure Blender to use GPU rendering if available.
    Adjust 'compute_device_type' based on your GPU (CUDA for NVIDIA, HIP/OPENCL for AMD).
    """
    bpy.context.scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'  # For NVIDIA. Use 'HIP' or 'OPENCL' for AMD.
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True


# Set the render engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'

# Enable GPU rendering once (if possible)
enable_gpu()

# Improve render settings
bpy.context.scene.cycles.samples = 1024  # Increase sample count for better quality
bpy.context.scene.cycles.use_adaptive_sampling = True
bpy.context.scene.cycles.use_denoising = True  # Enable denoising
bpy.context.scene.cycles.use_caustics_reflective = True
bpy.context.scene.cycles.use_caustics_refractive = True

# Adjust light path settings for more realistic ray tracing
bpy.context.scene.cycles.max_bounces = 12
bpy.context.scene.cycles.diffuse_bounces = 4
bpy.context.scene.cycles.glossy_bounces = 4
bpy.context.scene.cycles.transparent_max_bounces = 8
bpy.context.scene.cycles.transmission_bounces = 12
bpy.context.scene.cycles.volume_bounces = 2

# Disable film transparency
bpy.context.scene.render.film_transparent = False

# Set the output image format to PNG and color mode to RGB
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGB'

# Number of random perspectives to render
num_perspectives = 5

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Collect all .gltf files in the models directory
gltf_files = glob.glob(os.path.join(models_directory, '*.gltf'))

# Process each .gltf file
for gltf_file in gltf_files:
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(gltf_file))[0]

    # Create a subdirectory for the current GLTF file
    gltf_output_dir = os.path.join(output_directory, file_name)
    if not os.path.exists(gltf_output_dir):
        os.makedirs(gltf_output_dir)

    # Clear the scene before importing the model
    clear_scene()

    # Reset the white background after clearing the scene
    set_white_background()

    # Ensure GPU is still enabled for rendering
    enable_gpu()

    # Import the GLTF file
    bpy.ops.import_scene.gltf(filepath=gltf_file)

    # Select the main mesh object
    if bpy.context.selected_objects:
        mesh_objects = [
            obj for obj in bpy.context.selected_objects
            if isinstance(obj.data, bpy.types.Mesh)
        ]
        if not mesh_objects:
            raise Exception(f"No mesh object found in {gltf_file}")
        main_obj = mesh_objects[0]
    else:
        raise Exception(f"No object imported from {gltf_file}")

    # Move object to origin and set origin to its geometry
    bpy.context.view_layer.objects.active = main_obj
    main_obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    main_obj.location = (0, 0, 0)

    # Apply scale transformations
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Update mesh data
    main_obj.data.update()

    print(f"Imported object: {main_obj.name}")
    print(f"Location: {main_obj.location}")
    print(f"Dimensions: {main_obj.dimensions}")

    # Ensure the main object has valid mesh data
    if not isinstance(main_obj.data, bpy.types.Mesh):
        raise Exception(f"Imported object '{main_obj.name}' has no mesh data.")

    # Calculate the model size and the maximum dimension
    size_x, size_y, size_z = main_obj.dimensions
    max_dimension = max(size_x, size_y, size_z)

    # Setup Freestyle for outline rendering
    setup_freestyle(max_dimension)

    # Determine camera distance based on the model size
    distance = max_dimension * 2.75

    # Create and link a new camera
    camera_data = bpy.data.cameras.new("Camera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj

    # Adjust camera clipping
    camera_obj.data.clip_start = 0.1
    camera_obj.data.clip_end = distance * 10

    # Optionally adjust the focal length
    camera_obj.data.lens = 50

    # Disable depth of field
    camera_obj.data.dof.use_dof = False

    # Add a light source from above and to one side
    light_data = bpy.data.lights.new(name="SunLight", type='SUN')
    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    # Dynamically place the light (above and to the side)
    light_distance = max_dimension * 5
    light_obj.location = (light_distance, -light_distance, light_distance)
    light_data.energy = 1
    light_data.use_shadow = True
    light_data.cycles.use_multiple_importance_sampling = True

    # Orient the light towards the main object
    direction = main_obj.location - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Soften shadows by increasing the filter width
    bpy.context.scene.cycles.filter_width = 1.5

    # Ensure the object can cast and receive shadows
    main_obj.visible_shadow = True

    # Check and configure materials on the main object
    if main_obj.data.materials:
        for mat in main_obj.data.materials:
            mat.use_nodes = True
            # Ensure the material has a Principled BSDF node
            if not mat.node_tree.nodes.get('Principled BSDF'):
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                nodes.clear()
                principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
    else:
        # If no material is found, create a default one
        default_mat = bpy.data.materials.new(name="DefaultMaterial")
        default_mat.use_nodes = True
        nodes = default_mat.node_tree.nodes
        nodes.clear()
        principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.5
        nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
        default_mat.node_tree.links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
        main_obj.data.materials.append(default_mat)

    # Set rendering resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    # --- Render random perspectives ---
    for i in range(num_perspectives):
        # Generate random spherical coordinates
        phi = random.uniform(0, 2 * math.pi)      # Azimuth angle
        theta = random.uniform(0, math.pi / 2)    # Polar angle (upper hemisphere)

        # Convert spherical to Cartesian
        x = distance * math.sin(theta) * math.cos(phi)
        y = distance * math.sin(theta) * math.sin(phi)
        z = distance * math.cos(theta)

        camera_obj.location = (x, y, z)

        # Align camera to look at the main object
        align_camera(camera_obj, main_obj.location)

        # Set the output path
        output_file = os.path.join(
            gltf_output_dir, f"{file_name}_random_{i + 1:02d}.png"
        )
        bpy.context.scene.render.filepath = output_file

        # Render and save the image
        bpy.ops.render.render(write_still=True)
