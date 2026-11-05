"""Write a RoboLab scene USD from a JSON spec (robot base frame: +x forward, +y left, z up, base at origin).

Children of /world become scene entities by name. Prims whose USD has an enabled RigidBodyAPI (objects
from make_asset.py) are movable objects; everything else (surfaces, fixtures, floor, lights) is static.
Asset paths are relative to the output file (scenes live in RoboLab/assets/scenes/ -> "../objects/...").

spec.json (every list may be empty; nothing is required except "out"):
{"out": "RoboLab/assets/scenes/my_task.usda",
 "surfaces": [
   {"name": "table", "shape": "box", "pos": [0.40, 0.08, 0.0], "size": [0.70, 0.65, 0.70], "yaw_deg": 0,
    "material": {"texture": "../my_task_scene/top.png", "stretch": true}, "physics": {"static_friction": 0.6}},
   {"name": "tray", "shape": "cylinder", "pos": [0.5, -0.2, 0.02], "size": [0.30, 0.30, 0.02],
    "material": {"color": [0.8, 0.8, 0.8], "roughness": 0.3}},
   {"name": "mat", "shape": "box", "pos": [0.45, 0.0, 0.004], "size": [0.40, 0.30, 0.004],
    "material": {"texture": "../my_task_scene/mat_tile.png", "period_m": [0.12, 0.12]},
    "drape": {"drop": 0.0}},
   {"name": "shelf", "shape": "usd", "usd": "../fixtures/my_shelf.usd", "pos": [0.9, 0.5, -0.7], "yaw_deg": 90,
    "scale": [1, 1, 1]}],
 "floor": {"z": -0.75, "material": {"color": [0.11, 0.115, 0.125]}},
 "lights": [{"type": "RectLight", "pos": [0.4, 0.0, 1.6], "intensity": 3000, "color": [1, 0.97, 0.92],
             "width": 1.2, "height": 0.3}],
 "objects": [{"name": "obj_a", "usd": "../objects/my_task/obj_a/obj_a.usda", "pos": [0.45, -0.08, 0.005],
              "yaw_deg": 30}]}
box / cylinder: "pos" is the centre of the TOP face, "size" = [x, y, height] ([diameter, diameter, height]
  for cylinders). They are UV-mapped meshes with static collision.
material: {"color"} or {"texture" + "period_m": [u, v]} (tiled at real metric period) or {"texture" +
  "stretch": true} (one rectified photo spans the top face; sides tile at the same scale). Optional
  "roughness", "metallic". "physics" adds a physics material (static/dynamic friction, restitution).
drape: a thin textured skin over the top and down the sides of a box (cloth, mat, paper); keys "drop",
  "overhang", "material" (defaults to the surface's material).
usd: any fixture USD as a payload (static unless it has an enabled RigidBodyAPI), with "pos", "yaw_deg" or
  "quat_wxyz", "scale".
Floor defaults to 2 mm below the lowest surface bottom (or z = -0.75 without surfaces).

Usage: python gen_scene.py spec.json
"""
import json
import math
import os
import sys


def f3(v):
    return "(" + ", ".join(f"{c:.4f}" for c in v) + ")"


def quat(o):
    if "quat_wxyz" in o:
        return o["quat_wxyz"]
    y = math.radians(o.get("yaw_deg", 0.0))
    return (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))


def xform(pos, q=(1, 0, 0, 0), scale=(1, 1, 1), ind="        "):
    return (f"{ind}quatd xformOp:orient = ({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})\n"
            f"{ind}double3 xformOp:scale = {f3(scale)}\n{ind}double3 xformOp:translate = {f3(pos)}\n"
            f'{ind}uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]')


def material(name, m):
    rough, metal = m.get("roughness", 0.8), m.get("metallic", 0.0)
    if "texture" not in m:
        return f'''
    def Material "{name}"
    {{
        token outputs:surface.connect = </world/{name}/Shader.outputs:surface>
        def Shader "Shader"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = {f3(m.get("color", (0.5, 0.5, 0.5)))}
            float inputs:roughness = {rough}
            float inputs:metallic = {metal}
            token outputs:surface
        }}
    }}'''
    return f'''
    def Material "{name}"
    {{
        token outputs:surface.connect = </world/{name}/Shader.outputs:surface>
        def Shader "Shader"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </world/{name}/Texture.outputs:rgb>
            float inputs:roughness = {rough}
            float inputs:metallic = {metal}
            token outputs:surface
        }}
        def Shader "Texture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{m["texture"]}@
            token inputs:sourceColorSpace = "sRGB"
            float2 inputs:st.connect = </world/{name}/Reader.outputs:result>
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float3 outputs:rgb
        }}
        def Shader "Reader"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            string inputs:varname = "st"
            float2 outputs:result
        }}
    }}'''


def physics_material(name, ph):
    return f'''
    def Material "{name}" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {ph.get("static_friction", 0.6)}
        float physics:dynamicFriction = {ph.get("dynamic_friction", 0.5)}
        float physics:restitution = {ph.get("restitution", 0.0)}
    }}'''


def uv_scale(m, sx, sy):
    if m.get("stretch"):
        return (sx, sy)
    return tuple(m.get("period_m", (1.0, 1.0)))


def box_quads(sx, sy, h, drop=None, over=0.0):
    """Top face (z = 0) plus sides down to -h (or -drop), in local coords centred on the top face."""
    a0, a1, b0, b1 = -sx / 2 - over, sx / 2 + over, -sy / 2 - over, sy / 2 + over
    zb = -(h if drop is None else drop)
    d = -zb
    q = [([(a0, b0, 0), (a1, b0, 0), (a1, b1, 0), (a0, b1, 0)], [(a0, b0), (a1, b0), (a1, b1), (a0, b1)])]
    if d > 0:
        q += [([(a0, b1, zb), (a0, b0, zb), (a0, b0, 0), (a0, b1, 0)], [(a0 - d, b1), (a0 - d, b0), (a0, b0), (a0, b1)]),
              ([(a1, b0, zb), (a1, b1, zb), (a1, b1, 0), (a1, b0, 0)], [(a1 + d, b0), (a1 + d, b1), (a1, b1), (a1, b0)]),
              ([(a0, b0, zb), (a1, b0, zb), (a1, b0, 0), (a0, b0, 0)], [(a0, b0 - d), (a1, b0 - d), (a1, b0), (a0, b0)]),
              ([(a1, b1, zb), (a0, b1, zb), (a0, b1, 0), (a1, b1, 0)], [(a1, b1 + d), (a0, b1 + d), (a0, b1), (a1, b1)])]
        if drop is None:
            q.append(([(a0, b0, zb), (a0, b1, zb), (a1, b1, zb), (a1, b0, zb)], [(a0, b0), (a0, b1), (a1, b1), (a1, b0)]))
    return q


def cyl_quads(dia, h, n=48):
    r = dia / 2
    ang = [2 * math.pi * i / n for i in range(n + 1)]
    q = []
    for i in range(n):
        c0, s0, c1, s1 = math.cos(ang[i]), math.sin(ang[i]), math.cos(ang[i + 1]), math.sin(ang[i + 1])
        q.append(([(0, 0, 0), (r * c0, r * s0, 0), (r * c1, r * s1, 0), (0, 0, 0)], [(0, 0), (r * c0, r * s0), (r * c1, r * s1), (0, 0)]))
        u0, u1 = r * ang[i], r * ang[i + 1]
        q.append(([(r * c0, r * s0, -h), (r * c1, r * s1, -h), (r * c1, r * s1, 0), (r * c0, r * s0, 0)], [(u0, -h), (u1, -h), (u1, 0), (u0, 0)]))
        q.append(([(0, 0, -h), (r * c1, r * s1, -h), (r * c0, r * s0, -h), (0, 0, -h)], [(0, 0), (r * c1, r * s1), (r * c0, r * s0), (0, 0)]))
    return q


def mesh(name, quads, uvs_scale, offset_uv, mat, pos, q, collide, phys):
    su, sv = uvs_scale
    ou, ov = offset_uv
    pts = [p for qd in quads for p in qd[0]]
    uvs = [((u + ou) / su, (v + ov) / sv) for qd in quads for u, v in qd[1]]
    api = '(\n        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]\n    )' if collide else ""
    coll = '\n        uniform token physics:approximation = "none"' if collide else ""
    pb = f"\n        rel material:binding:physics = </world/{phys}>" if phys else ""
    return f'''
    def Mesh "{name}" {api}
    {{
        uniform bool doubleSided = 1
        int[] faceVertexCounts = [{", ".join("4" for _ in quads)}]
        int[] faceVertexIndices = [{", ".join(str(i) for i in range(len(pts)))}]
        point3f[] points = [{", ".join(f3(p) for p in pts)}]
        texCoord2f[] primvars:st = [{", ".join(f"({u:.4f}, {v:.4f})" for u, v in uvs)}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </world/{mat}>{coll}{pb}
{xform(pos, q)}
    }}'''


def main():
    spec = json.load(open(sys.argv[1]))
    parts, bottoms = [], []
    for s in spec.get("surfaces", []):
        n, shape = s["name"], s.get("shape", "box")
        q = quat(s)
        if shape == "usd":
            parts.append(f'''
    def "{n}" (
        prepend payload = @{s["usd"]}@
    )
    {{
{xform(s.get("pos", (0, 0, 0)), q, s.get("scale", (1, 1, 1)))}
    }}''')
            continue
        sx, sy, h = s["size"]
        bottoms.append(s["pos"][2] - h)
        m = s.get("material", {"color": (0.5, 0.5, 0.5)})
        parts.append(material(f"{n}_look", m))
        phys = None
        if "physics" in s:
            phys = f"{n}_phys"
            parts.append(physics_material(phys, s["physics"]))
        quads = box_quads(sx, sy, h) if shape == "box" else cyl_quads(sx, h)
        off = (sx / 2, sy / 2) if m.get("stretch") else (0.0, 0.0)
        parts.append(mesh(n, quads, uv_scale(m, sx, sy), off, f"{n}_look", s["pos"], q, s.get("collision", True), phys))
        if "drape" in s and shape == "box":
            d = s["drape"]
            dm = d.get("material", m)
            if dm is not m:
                parts.append(material(f"{n}_drape_look", dm))
            p = (s["pos"][0], s["pos"][1], s["pos"][2] + d.get("lift", 0.002))
            parts.append(mesh(f"{n}_drape", box_quads(sx, sy, h, d.get("drop", 0.2), d.get("overhang", 0.01)),
                              uv_scale(dm, sx, sy), off, f"{n}_drape_look" if dm is not m else f"{n}_look", p, q, False, None))
    fl = spec.get("floor", {})
    if fl is not None:
        z = fl.get("z", (min(bottoms) - 0.002) if bottoms else -0.75)
        parts.append(material("floor_look", fl.get("material", {"color": (0.11, 0.115, 0.125), "roughness": 0.6})))
        parts.append(f'''
    def Mesh "floor"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-4, -4, {z:.4f}), (4, -4, {z:.4f}), (4, 4, {z:.4f}), (-4, 4, {z:.4f})]
        texCoord2f[] primvars:st = [(-4, -4), (4, -4), (4, 4), (-4, 4)] (
            interpolation = "vertex"
        )
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </world/floor_look>
    }}

    def Xform "GroundPlane"
    {{
        token visibility = "invisible"
{xform((0, 0, z))}

        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }}
    }}''')
    for i, L in enumerate(spec.get("lights", [])):
        extra = "".join(f"\n        float inputs:{k} = {L[k]}" for k in ("width", "height", "radius", "angle") if k in L)
        parts.append(f'''
    def {L.get("type", "RectLight")} "{L.get("name", f"light_{i}")}"
    {{
        float inputs:intensity = {L["intensity"]}
        color3f inputs:color = {f3(L.get("color", (1, 1, 1)))}{extra}
        quatf xformOp:orient = {f3(L.get("quat_wxyz", (1, 0, 0, 0)))}
        double3 xformOp:translate = {f3(L.get("pos", (0, 0, 2)))}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}''')
    for o in spec.get("objects", []):
        parts.append(f'''
    def "{o["name"]}" (
        prepend payload = @{o["usd"]}@
    )
    {{
{xform(o["pos"], quat(o), o.get("scale", (1, 1, 1)))}
    }}''')
    usda = f'''#usda 1.0
(
    defaultPrim = "world"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)

def Xform "world"
{{
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
{"".join(parts)}
}}
'''
    os.makedirs(os.path.dirname(os.path.abspath(spec["out"])), exist_ok=True)
    open(spec["out"], "w").write(usda)
    print("wrote", spec["out"], "surfaces", [s["name"] for s in spec.get("surfaces", [])],
          "objects", [o["name"] for o in spec.get("objects", [])])


if __name__ == "__main__":
    main()
