import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import gym
import pygame

import cma_es
import main
from keypoints.models import transporter
import config
from utils import UniImageViewer
import torch
from pyrr import matrix44, Vector3
from math import floor
from time import sleep
from PIL import Image

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


fragment_bilinear_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

vec2 textureSize = vec2(1.0, 1.0);
vec2 texelSize = vec2(1/160, 1/210);

vec4 texture2D_bilinear(in sampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize)
{
    vec2 f = fract( uv * textureSize );
    uv += ( .5 - f ) * texelSize;    // move uv to texel centre
    vec4 tl = texture2D(t, uv);
    vec4 tr = texture2D(t, uv + vec2(texelSize.x, 0.0));
    vec4 bl = texture2D(t, uv + vec2(0.0, texelSize.y));
    vec4 br = texture2D(t, uv + vec2(texelSize.x, texelSize.y));
    vec4 tA = mix( tl, tr, f.x );
    vec4 tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}

void main() {
    out_color = texture2D_bilinear(s_texture, v_texture, textureSize, texelSize);
}
"""

fragment_maxpool_nowork_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

vec2 textureSize = vec2(1.0, 1.0);
vec2 texelSize = vec2(0.01, 0.01);

vec4 texture2D_bilinear(in sampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize)
{
    vec2 f = fract( uv * textureSize );
    uv += ( .5 - f ) * texelSize;    // move uv to texel centre
    vec4 tl = texture2D(t, uv);
    vec4 tr = texture2D(t, uv + vec2(texelSize.x, 0.0));
    vec4 bl = texture2D(t, uv + vec2(0.0, texelSize.y));
    vec4 br = texture2D(t, uv + vec2(texelSize.x, texelSize.y));
    vec4 tA = max( tl, tr);
    vec4 tB = max( bl, br);
    
    return max( tA, tB );
}

void main() {
    out_color = texture2D_bilinear(s_texture, v_texture, textureSize, texelSize);
}
"""


fragment_maxpool_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

vec2 textureSize = vec2(1.0, 1.0);
vec2 texelSize = vec2(0.01, 0.01);

vec4 texture2D_maxpool(in sampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize)
{
    vec2 f = fract( uv * textureSize );
    uv += ( .5 - f ) * texelSize;    // move uv to texel centre
    
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 tl;
    
    for (int i =-1; i < 3; i++) {
      for (int j =-1; j < 3; j++) {
        tl = texture(t, uv + vec2(texelSize.x * i, texelSize.y * j));
        color = max( tl, color );
      }
    } 
    
    return color;
}

void main() {
    out_color = texture2D_maxpool(s_texture, v_texture, textureSize, texelSize);
}
"""


compileShader(fragment_bilinear_src, GL_FRAGMENT_SHADER)

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

#shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_bilinear_src, GL_FRAGMENT_SHADER))

maxpool_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_maxpool_src, GL_FRAGMENT_SHADER))

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

translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]))
atari_scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([float(atari_width), float(atari_height), 1.0]))
atari_screen1_model = pyrr.matrix44.multiply(atari_scale, translation)

zoom = 2.2
atari_screen2_scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([float(atari_width) * zoom, float(atari_height) * zoom, 1.0]))
atari_screen2_translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]))
atari_screen_2 = pyrr.matrix44.multiply(atari_screen2_scale, atari_screen2_translation)

translation_cut = matrix44.create_from_translation(Vector3([0.0, -34.0, 0.0]))
scale_cut = matrix44.create_from_scale(Vector3([160, 210, 1.0]))
cutout_model = pyrr.matrix44.multiply(scale_cut, translation_cut)

minime_translate = matrix44.create_from_translation(Vector3([0.0, -34.0 / 210.0, 0.0]))
minime_scale = matrix44.create_from_scale(Vector3([32.0, (210.0 / (168.0 - 34.0)) * 32.0, 1.0]))
minime_model = matrix44.multiply(minime_translate, minime_scale)
minime_inv_model = matrix44.inverse(minime_model)
transporter_scale = matrix44.create_from_scale(Vector3([32.0, 32.0, 1.0]))

bbox_scale_factor = 16.0/0.9
bbox1_model = pyrr.matrix44.create_from_scale(pyrr.Vector3([bbox_scale_factor, bbox_scale_factor, 1.0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")

color_model_loc = glGetUniformLocation(white_shader, "model")
color_proj_loc = glGetUniformLocation(white_shader, "projection")
color_loc = glGetUniformLocation(white_shader, "uColor")


# init env
env = gym.make('Pong-v0')
env.reset()

# init transporter keypoint network
args = config.config(['--config', '../configs/cma_es/exp2/baseline.yaml'])
datapack = keypoints.ds.datasets.datasets[args.dataset]
transporter_net = transporter.make(args, map_device='cpu')
view = main.Keypoints(transporter_net)


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


class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pyrr_orthogonal_proj = -width/2.0, width/2.0, -height/2.0, height/2.0, -1000.0, 1000


anchor = {'source': Pos(0, 0),
           'plot': Pos(0, atari_height + 10),
           'kp_input': Pos(atari_width + 10, 0),
           'scratch': Pos(atari_width * 2 + 80, 400),
           }

xpos, ypos = None, None

imageid = 0

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    xpos, ypos = glfw.get_cursor_pos(window)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # take a step in the environment
    image_data, r, done, info = env.step(cma_es.sample())
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[1], image_data.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE,
                 image_data)
    glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    if done:
        env.reset()

    # render reference screen used for sampling
    glViewport(anchor['source'].x, anchor['source'].y, atari_width, atari_height)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, atari_width, 0, atari_height, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, atari_screen1_model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # kp_input model
    glUseProgram(maxpool_shader)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 32, 0, 32, -1000, 1000)
    glViewport(anchor['kp_input'].x, anchor['kp_input'].y, 32, 32)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, minime_model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # preprocessed input,sent to the keypoint network here
    buffer = glReadPixels(anchor['kp_input'].x, anchor['kp_input'].y, 32, 32, format=GL_RGB, type=GL_UNSIGNED_BYTE)
    pixel_array = np.frombuffer(buffer, dtype=np.uint8).reshape(32, 32, 3)
    #viewer.render(pixel_array)
    with torch.no_grad():
        s_t = datapack.transforms(pixel_array).unsqueeze(0)
        kp = view(s_t)

    # convert keypoints to basis space
    kp_n = kp.detach().cpu().numpy()[0]
    kp_n = np.roll(kp_n, axis=1, shift=1)
    kp_n = np.concatenate((kp_n, np.ones((kp_n.shape[0], 2))), axis=1)
    kp_n = matrix44.multiply(kp_n, transporter_scale)
    kp_n = matrix44.multiply(kp_n, minime_inv_model)

    # plotting screen
    glUseProgram(shader)
    glViewport(anchor['plot'].x, anchor['plot'].y, floor(atari_width * zoom),  floor((atari_height * zoom)))
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, atari_width * zoom, 0, atari_height * zoom, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, atari_screen_2)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # draw the bounding boxes on kp locations
    glUseProgram(white_shader)

    for i, k in enumerate(kp_n):
        glUniformMatrix4fv(color_proj_loc, 1, GL_FALSE, projection)
        model_translate = pyrr.matrix44.create_from_translation(Vector3([k[0], k[1], 2.0]))
        model_scale = pyrr.matrix44.create_from_scale(Vector3([8.0/(atari_width * 0.9), 8.0/(atari_height * 0.9), 1.0]))
        model = matrix44.multiply(model_scale, model_translate)
        model = matrix44.multiply(model, atari_screen_2)
        glUniformMatrix4fv(color_model_loc, 1, GL_FALSE, model)
        glUniform4fv(color_loc, 1, cmap[i])
        glDrawElements(GL_QUADS, offsets['bbox'].len, GL_UNSIGNED_INT, offsets['bbox'].offset)

    glUseProgram(shader)

    # scaled up view of the NN input view
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 160, 34, 168, -1000, 1000)
    glViewport(anchor['scratch'].x, anchor['scratch'].y, 256, 256)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, atari_screen1_model)
    glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    # draw bounding boxes on NN view
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

    glUseProgram(shader)

    # draw labels for key value pairs
    glViewport(anchor['scratch'].x, anchor['scratch'].y - 256, 265, 256)
    placeholder = np.ones((3, 32, 32), dtype=np.uint8) * 255

    y_align = 0.7
    for k in kp[0, :, :]:
        key = 'key'
        label = f"{key} x: {k[0].item():.3f} y: {k[1].item():.3f}"
        drawText((-0.5, y_align, 0), label, fontsize=24)
        y_align -= 0.4

    for i, k in enumerate(kp_n):
        anchr = Pos(anchor['scratch'].x, anchor['scratch'].y - (i * 50) - 50)
        glViewport(anchr.x, anchr.y, 32, 32)
        model_trans = matrix44.create_from_translation(Vector3((-k[0], -k[1], 0.0)))
        model_scale = matrix44.create_from_scale(Vector3((atari_width, atari_height, 1.0)))
        model = matrix44.multiply(model_trans, model_scale)
        projection = pyrr.matrix44.create_orthogonal_projection_matrix(-8.0, 8.0, -8.0, 8.0, -1000, 1000)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

    for i, k in enumerate(kp_n):
        anchr = Pos(anchor['kp_input'].x, anchor['kp_input'].y + (i * 50) + 50)
        size = Size(16, 16)
        glViewport(anchr.x, anchr.y, size.width, size.height)
        model_trans = matrix44.create_from_translation(Vector3((-k[0], -k[1], 0.0)))
        model_scale = matrix44.create_from_scale(Vector3((atari_width, atari_height, 1.0)))
        model = matrix44.multiply(model_trans, model_scale)
        projection = pyrr.matrix44.create_orthogonal_projection_matrix(*size.pyrr_orthogonal_proj)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawElements(GL_TRIANGLES, offsets['screen'].len, GL_UNSIGNED_INT, offsets['screen'].offset)

        buffer = glReadPixels(anchr.x, anchr.y, size.width, size.height, format=GL_RGB, type=GL_UNSIGNED_BYTE)
        image = Image.frombuffer(mode='RGB', size=(size.width, size.height), data=buffer)
        image.save(f'/home/duane/PycharmProjects/keypoints/data/patches/pong/unclassified/pong_{imageid}.png')
        pixel_array = np.frombuffer(buffer, dtype=np.uint8).reshape(size.width, size.height, 3)
        viewer.render(pixel_array)
        imageid += 1

    glfw.swap_buffers(window)

    sleep(0.01)

# terminate glfw, free up allocated resources
glfw.terminate()
