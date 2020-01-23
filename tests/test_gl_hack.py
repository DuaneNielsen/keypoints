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

vertices += [
    0.0, 0.0, 0.0, 0.0
]

indices = [0,  1,  2,  2,  3,  0]


class Offset:
    def __init__(self, offset, len):
        self.offset = ctypes.c_void_p(offset*ctypes.sizeof(ctypes.c_int))
        self.len = len


offsets = {'screen': Offset(0, 6)
           }

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

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

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[1], image_data.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE,
                 image_data)
    if done:
        env.reset()

    anchor_x, anchor_y = 0, 0
    glViewport(anchor_x, anchor_y, (atari_width * 2) + border, atari_height)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, atari_width * 2 + border, 0, atari_height, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, screen_2)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)


    anchor_x += 200 * 2
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(anchor_x, 0, 160, 134)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    anchor_x += 200
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(anchor_x, 0, 32, 32)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    drawText((0, 0, 0), 'HELLO', fontsize=12)

    pixel = glReadPixels(anchor_x, 0, 32, 32, format=GL_RGB, type=GL_UNSIGNED_BYTE)
    pixel_array = np.frombuffer(pixel, dtype=np.uint8).reshape(3, 32, 32)
    #drawNumpy((0, 0, 0), np.ones((3, 32, 32), dtype=np.uint8) * 244)

    #s_t = datapack.transforms(pixel_array).unsqueeze(0)
    #kp = view(s_t)

    glViewport(0, 0, window_width, window_height)
    #drawText((0, 0, 0), 'HELLO', fontsize=12)
    drawNumpy((0, 0, 0), pixel_array)

    anchor_x += 200
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(anchor_x, 0, 256, 256)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    glfw.swap_buffers(window)



# terminate glfw, free up allocated resources
glfw.terminate()
