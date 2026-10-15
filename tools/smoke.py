import mujoco, numpy as np

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(500):
    mujoco.mj_step(model, data)
print("box z after 1s of falling:", data.qpos[2])

renderer = mujoco.Renderer(model, 240, 320)
renderer.update_scene(data)
print("frame shape:", renderer.render().shape)