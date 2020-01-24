import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import gym
import pygame
import cma_es
from models import transporter
import config
import ds
from utils import UniImageViewer
import torch
from pyrr import matrix44, Vector3, Vector4

viewer = UniImageViewer()

pygame.init()

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 model; // combined translation and rotation
uniform mat4 projection;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = projection * model * vec4(a_position, 1.0);
    //v_texture = 1 - a_texture; 
    v_texture = vec2(a_texture.s, 1 - a_texture.t);;
}
"""

fragment_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
}
"""


vertex_white_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 model; // combined translation and rotation
uniform mat4 projection;

//out vec4 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = projection * model * vec4(a_position, 1.0);
    //v_texture = 1 - a_texture; 
    v_texture = vec2(a_texture.s, 1 - a_texture.t);
}
"""


frament_white_src = """
# version 330

in vec4 v_color;
uniform vec4 uColor;
out vec4 out_color;

void main()
{
    //out_color = vec4(1.0, 1.0, 0.0, 1.0);
    out_color = uColor;
}
"""

cmap = [(1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),

        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5)]
cmap = [pyrr.vector4.create(*c, 1.0) for c in cmap]

# glfw callback functions
def window_resize(window, width, height):
    pass
    # glViewport(0, 0, width, height)
    # projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, width, 0, height, -1000, 1000)
    # glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

window_width, window_height = 1280, 720

# creating the window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)


vertices = [ 0.0,  0.0,  0.5, 0.0, 0.0,
             1.0,  0.0,  0.5, 1.0, 0.0,
             1.0,  1.0,  0.5, 1.0, 1.0,
             0.0,  1.0,  0.5, 0.0, 1.0]

indices = [0,  1,  2,  2,  3,  0]


v = [-1.0,  -1.0,  0.5, 0.0, 0.0,
      1.0,  -1.0,  0.5, 1.0, 0.0,
      1.0,  1.0,  0.5, 1.0, 1.0,
     -1.0,  1.0,  0.5, 0.0, 1.0,

    -0.9,  -0.9,  0.5, 0.0, 0.0,
      0.9,  -0.9,  0.5, 1.0, 0.0,
      0.9,  0.9,  0.5, 1.0, 1.0,
      -0.9,  0.9,  0.5, 0.0, 1.0]

vertices += v

i = [0, 1, 5, 4,
     5, 1, 2, 6,
     7, 6, 2, 3,
     0, 4, 7, 3]
i = [item + 4 for item in i]
indices += i



class Offset:
    def __init__(self, offset, len):
        self.offset = ctypes.c_void_p(offset*ctypes.sizeof(ctypes.c_int))
        self.len = len


offsets = {'screen': Offset(0, 6),
           'bbox': Offset(6, len(i))
           }

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

white_shader = compileProgram(compileShader(vertex_white_src, GL_VERTEX_SHADER), compileShader(frament_white_src, GL_FRAGMENT_SHADER))


# Vertex Buffer Object
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Element Buffer Object
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)

# Set the texture wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
# Set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# load image
# image = Image.open("textures/crate.jpg")
# image = image.transpose(Image.FLIP_TOP_BOTTOM)
# img_data = image.convert("RGBA").tobytes()
# img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data
# glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


atari_width, atari_height = 160, 210
border = 10

# projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280/720, 0.1, 100)
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]))
scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([float(atari_width), float(atari_height), 1.0]))
model = pyrr.matrix44.multiply(scale, translation)

translation_screen2 = pyrr.matrix44.create_from_translation(pyrr.Vector3([float(atari_width + border), 0, -3]))
screen_2 = pyrr.matrix44.multiply(scale, translation_screen2)

translation_cut = matrix44.create_from_translation(Vector3([0.0, -34.0, 0.0]))
scale_cut = matrix44.create_from_scale(Vector3([160, 210, 1.0]))
cutout_model = pyrr.matrix44.multiply(scale_cut, translation_cut)

minime_transform = matrix44.create_from_translation(Vector3([0.0, -34.0/210.0, 0.0]))
minime_scale = matrix44.create_from_scale(Vector3([32.0, (210.0 / (168.0 - 34.0)) * 32.0, 1.0]))
minime_model = matrix44.multiply(minime_transform, minime_scale)

bbox_scale_factor = 16.0/0.9
bbox1_model = pyrr.matrix44.create_from_scale(pyrr.Vector3([bbox_scale_factor, bbox_scale_factor, 1.0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")

color_model_loc = glGetUniformLocation(white_shader, "model")
color_proj_loc = glGetUniformLocation(white_shader, "projection")
color_loc = glGetUniformLocation(white_shader, "uColor")


env = gym.make('Pong-v0')
env.reset()

args = config.config(['--config', '../configs/cma_es/exp2/baseline.yaml'])
datapack = ds.datasets.datasets[args.dataset]

transporter_net = transporter.make(args, map_device='cpu')
view = cma_es.Keypoints(transporter_net)

xpos, ypos = None, None

def drawText(position, textString, fontsize=128, forecolor=(255,255,255,255), backcolor=(0, 0, 0, 255)):
    shader = glGetInteger(GL_CURRENT_PROGRAM)
    glUseProgram(0)
    font = pygame.font.Font (None, fontsize)
    textSurface = font.render(textString, True, forecolor, backcolor)
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    raster_pos = glGetInteger(GL_CURRENT_RASTER_POSITION)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)
    glUseProgram(shader)

def drawNumpy(position, data):
    shader = glGetInteger(GL_CURRENT_PROGRAM)
    glUseProgram(0)
    glRasterPos3d(*position)
    raster_pos = glGetInteger(GL_CURRENT_RASTER_POSITION)
    glDrawPixels(data.shape[2], data.shape[1], GL_RGB, GL_UNSIGNED_BYTE, data)
    glUseProgram(shader)

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    xpos, ypos = glfw.get_cursor_pos(window)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    image_data, r, done, info = env.step(env.action_space.sample())
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[1], image_data.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE,
                 image_data)
    if done:
        env.reset()

    # render 2 copies of the atari screen, for source and plotting
    anchor_x, anchor_y = 0, 0
    glViewport(anchor_x, anchor_y, (atari_width * 2) + border, atari_height)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, atari_width * 2 + border, 0, atari_height, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, screen_2)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # anchor_x += 200 * 2
    # glViewport(anchor_x, anchor_y, atari_width, 168 - 34)
    # projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, atari_width, 0, 168-34, -1000, 1000)
    # glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    # glUniformMatrix4fv(model_loc, 1, GL_FALSE, cutout_model)
    # glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # only focus on the actual game area, ignore the score
    # anchor_x += 200
    # projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    # glViewport(anchor_x, anchor_y, 160, 134)
    # glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    # glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    # glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # save view as above but scaled to 32, 32
    # anchor_x += 200
    # projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    # glViewport(anchor_x, anchor_y, 32, 32)
    # glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    # glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    # glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # minime model
    anchor_x += 200 * 2
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 32, 0, 32, -1000, 1000)
    glViewport(anchor_x, anchor_y, 32, 32)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, minime_model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # preprocessed input,sent to the keypoint network here
    pixel = glReadPixels(anchor_x, anchor_y, 32, 32, format=GL_RGB, type=GL_UNSIGNED_BYTE)
    pixel_array = np.frombuffer(pixel, dtype=np.uint8).reshape(32, 32, 3)
    #viewer.render(pixel_array)
    with torch.no_grad():
        s_t = datapack.transforms(pixel_array).unsqueeze(0)
        kp = view(s_t)




    # scaled up view of the preprocessed view
    anchor_x += 200
    anchor_y += 400
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(anchor_x, anchor_y, 256, 256)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    glUseProgram(white_shader)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 256, 0, 256, -1000, 1000)

    for i, k in enumerate(kp[0, :, :]):
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([k[1].item() * 16 * 0.9,
                                                                          k[0].item() * 16 * 0.9, 1.0]))
        m = pyrr.matrix44.multiply(translation, bbox1_model)
        glUniformMatrix4fv(color_proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(color_model_loc, 1, GL_FALSE, m)
        glUniform4fv(color_loc, 1, cmap[i])
        glDrawElements(GL_QUADS, offsets['bbox'].len, GL_UNSIGNED_INT, offsets['bbox'].offset)

    glViewport(atari_width, 0, atari_width + border, atari_height)
    projection = pyrr.matrix44.inverse(projection)

    for i, k in enumerate(kp[0, :, :]):
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([k[1].item()  * 0.9,
                                                                          k[0].item()  * 0.9, 1.0]))
        m = pyrr.matrix44.multiply(translation, bbox1_model)
        glUniformMatrix4fv(color_proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(color_model_loc, 1, GL_FALSE, m)
        glUniform4fv(color_loc, 1, cmap[i])
        glDrawElements(GL_QUADS, offsets['bbox'].len, GL_UNSIGNED_INT, offsets['bbox'].offset)

    glUseProgram(shader)

    # draw key value pairs
    anchor_y -= 256
    glViewport(anchor_x, anchor_y, 265, 256)
    placeholder = np.ones((3, 32, 32), dtype=np.uint8) * 255

    y_align = 0.7
    for k in kp[0, :, :]:
        key = 'key'
        label = f"{key} x: {k[0].item():.3f} y: {k[1].item():.3f}"
        drawText((-0.5, y_align, 0), label, fontsize=24)
        drawNumpy((-1.0, y_align, 0), placeholder)
        y_align -= 0.4




    # testing viewport
    # glUseProgram(white_shader)
    # glViewport(0, 400, 200, 200)
    # projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 128, 0, 128, -1000, 1000)
    # glUniformMatrix4fv(color_proj_loc, 1, GL_FALSE, projection)
    # glUniformMatrix4fv(color_model_loc, 1, GL_FALSE, bbox1_model)
    # color = pyrr.vector4.create(1.0, 1.0, 1.0, 1.0)
    # glUniform4fv(color_loc, 1, color)
    # glDrawElements(GL_QUADS, offsets['bbox'].len, GL_UNSIGNED_INT, offsets['bbox'].offset)
    # glUseProgram(shader)

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
