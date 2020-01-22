import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import gym
import torch
import ds.datasets
import torch
import gym
import gym_wrappers
import cma_es
from models import transporter
import config
from matplotlib import pyplot as plt


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


def make_rectangle(w, h, z=0.0, u=1.0, v=1.0):
    v = [0.0, 0.0, z, 0.0, 0.0,
         w, 0.0, z, u, 0.0,
         w, h, z, u, v,
         0.0, h, z, 0.0, v]
    i = [0,  1,  2,  2,  3,  0]
    v = np.array(v, dtype=np.float32)
    i = np.array(i, dtype=np.uint32)
    return v, i


vertices, indices = make_rectangle(160, 210)

m_v, m_i = make_rectangle(16, 16, 1.0)

#
# vertices = [-0.5, -0.5,  0.5, 0.0, 0.0,
#              0.5, -0.5,  0.5, 1.0, 0.0,
#              0.5,  0.5,  0.5, 1.0, 1.0,
#             -0.5,  0.5,  0.5, 0.0, 1.0]
#
# indices = [ 0,  1,  2,  2,  3,  0]

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

# projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280/720, 0.1, 100)
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]))
scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([1.0, 1.0, 1.0]))
model = pyrr.matrix44.multiply(scale, translation)

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")

env = gym.make('Pong-v0')
env.reset()

xpos, ypos = None, None

args = config.config(['--config', '../configs/cma_es/exp2/baseline.yaml'])
datapack = ds.datasets.datasets[args.dataset]

transporter_net = transporter.make(args, map_device='cpu')
view = cma_es.Keypoints(transporter_net)

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

    def draw():
        #rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        #rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())

        #rotation = pyrr.matrix44.multiply(rot_x, rot_y)
        #model = pyrr.matrix44.multiply(scale, rotation)

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glViewport(0, 0, 160, 210)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 0, 210, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    draw()

    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(200, 0, 160, 134)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    draw()

    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(400, 0, 32, 32)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    draw()

    pixel = glReadPixels(400, 0, 32, 32, format=GL_RGB, type=GL_UNSIGNED_BYTE)
    pixel_array = np.frombuffer(pixel, dtype=np.uint8).reshape(32, 32, 3)
    s_t = datapack.transforms(pixel_array).unsqueeze(0)
    kp = view(s_t)

    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(600, 0, 256, 256)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    draw()

    glfw.swap_buffers(window)



# terminate glfw, free up allocated resources
glfw.terminate()
