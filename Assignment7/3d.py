import pyglet
import ratcave as rc

window = pyglet.window.Window()

vet = open('vert.GLSL', 'r').read()

frag = open('fraq.GLSL', 'r').read()

def update(dt):
    pass
pyglet.clock.schedule(update)

obj_filename = rc.resources.obj_primitives
obj_reader = rc.WavefrontReader(obj_filename)

shader = rc.Shader(vet, frag)
print(obj_reader)
monkey = obj_reader.get_mesh("Sphere")
monkey.position.xyz = 0, 0, -3

#Uncomment to rotate sphere
""" def rotate_meshes(dt):
    monkey.rotation.y += 45 * dt
pyglet.clock.schedule(rotate_meshes) """

monkey.uniforms['ambientLightStreanght'] = 0.15
monkey.uniforms['color'] = [0.3, 0.0, 0.5]
monkey.uniforms['sunPosition'] = [1.0, -2.0, 3.0]


scene = rc.Scene(meshes=[monkey])
@window.event
def on_draw():
    with shader, rc.default_states:
        scene.draw()

pyglet.app.run()