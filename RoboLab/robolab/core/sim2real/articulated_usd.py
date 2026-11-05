# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an articulated object (lid, door, drawer, appliance) as a USD that a scene includes as a payload.

Links are primitive shapes joined by revolute or prismatic joints. "root" picks how the object is simulated:
- "fixed" (default): an articulation whose first link is fixed to the world (cabinets, wall units).
- "floating": an articulation whose first link is a dynamic body of mass "base_mass" (e.g. 30) resting on its
  support. Use it when the object's pose is randomized per env (fixtures.reset_fixture).
- "rigid_links": no articulation; the links are separate rigid bodies held by plain joints, the first one
  kinematic, or dynamic with "base_mass". Joint properties then cannot be changed at runtime; kept for
  older assets (an old spec's "fixed_base": true means this, false means "floating").
With "fixed" or "floating", register the object with fixtures.articulated_fixture_cfg; joint damping,
stiffness and friction then come from that config (SI units, per rad or per m) and can be randomized per env
(plan section "joints").

spec.json (object frame: origin on the support surface at the footprint centre, z up):
{"name": "waffle_maker", "root": "floating", "base_mass": 30, "out": ".../waffle_maker/waffle_maker.usda",
 "links": [
   {"name": "base", "mass": 1.5, "shapes": [
      {"type": "cylinder", "pos": [0, 0, 0.021], "radius": 0.098, "height": 0.042, "color": [0.03, 0.03, 0.035]},
      {"type": "box", "pos": [0, 0.11, 0.05], "size": [0.09, 0.055, 0.1], "yaw_deg": 0},
      {"type": "disc", "pos": [0, 0, 0.0425], "radius": 0.082, "collision": false}]},
   {"name": "lid", "mass": 0.4, "shapes": [
      {"type": "cylinder", "pos": [0, 0, 0.07], "radius": 0.104, "height": 0.055},
      {"type": "box", "name": "lip", "pos": [0, -0.115, 0.06], "size": [0.05, 0.025, 0.012]}]}],
 "joints": [{"name": "hinge", "type": "revolute", "parent": "base", "child": "lid", "anchor": [0, 0.12, 0.042],
             "axis": [-1, 0, 0], "limits": [0, 100]}],
 "friction": [0.7, 0.6]}
Shapes: cylinder (z axis; smooth visual mesh plus an implicit collision cylinder), prism (a convex [x, y]
polygon extruded over "z": [z0, z1], convex-hull collision), box (size x y z, optional yaw_deg), sphere, disc
(flat textured or coloured top, visual only by default). "color" [r g b] or "texture" (path relative to the
spec). Optional "roughness", "metallic", "collision" (default true except disc), "visible" (false makes a
collision-only proxy). Joints live under <name>/joints, so they may share a link's name. Shape and link
names must not repeat a scene object's name (contact-sensor patterns
match by name). A part the robot must grasp (handle, lip, knob) needs collision and must stand out from the
body by at least a finger's depth (~1.5 cm).
Joint "anchor"/"axis" are in the object frame; revolute "limits" in degrees, prismatic in metres. Positive
motion follows the right-hand rule about "axis". "damping" / "stiffness" in the spec are written to the USD
drive (USD units: per degree for revolute joints) and only matter for "rigid_links"; articulations take them
from articulated_fixture_cfg. Optional "filtered_pairs": [["lid", "base/control_block"], ...] turns off
collision between a link and a collision prim.

Usage: python -m robolab.core.sim2real.articulated_usd spec.json
"""
import json
import math
import os
import shutil
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def quat_from_x_to(axis):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    v, c = np.cross(x, a), float(np.dot(x, a))
    if np.linalg.norm(v) < 1e-9:
        return Gf.Quatf(1, 0, 0, 0) if c > 0 else Gf.Quatf(0, 0, 0, 1)
    s = math.sqrt((1 + c) * 2)
    return Gf.Quatf(s / 2, *(v / s))


def material(stage, path, sh, out_dir):
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(sh.get("roughness", 0.5)))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(sh.get("metallic", 0.0)))
    if "texture" in sh:
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(sh["texture"])
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*sh.get("color", [0.5, 0.5, 0.5])))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def disc_mesh(stage, path, radius, n=64):
    m = UsdGeom.Mesh.Define(stage, path)
    pts = [(0.0, 0.0, 0.0)] + [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * n))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([v for i in range(n) for v in (0, 1 + i, 1 + (i + 1) % n)]))
    uv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv.Set(Vt.Vec2fArray([Gf.Vec2f(0.5 + p[0] / (2 * radius), 0.5 + p[1] / (2 * radius)) for p in pts]))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(pts)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateDoubleSidedAttr(True)
    return m


def cylinder_mesh(stage, path, radius, height, n=96):
    """Smooth visual cylinder (implicit UsdGeom.Cylinder renders as a coarse polygon)."""
    m = UsdGeom.Mesh.Define(stage, path)
    h = height / 2
    ring = [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
    pts = [(x, y, -h) for x, y in ring] + [(x, y, h) for x, y in ring] + [(0, 0, -h), (0, 0, h)]
    counts, idx = [], []
    for i in range(n):
        j = (i + 1) % n
        counts.append(4); idx += [i, j, n + j, n + i]
        counts.append(3); idx += [2 * n, j, i]
        counts.append(3); idx += [2 * n + 1, n + i, n + j]
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr("none")
    return m


def prism_mesh(stage, path, poly, z0, z1):
    """Extruded convex polygon (counter-clockwise [x, y] list) from z0 to z1, with planar top UVs."""
    m = UsdGeom.Mesh.Define(stage, path)
    n = len(poly)
    pts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    counts, idx = [n, n], list(range(n))[::-1] + list(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4); idx += [i, j, n + j, n + i]
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr("none")
    return m


def build(spec, spec_dir="."):
    """Write the USD for spec (dict) to spec["out"]; texture paths are relative to spec_dir."""
    out = spec["out"]
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    for link in spec["links"]:
        for sh in link["shapes"]:
            if "texture" in sh:
                src = os.path.join(os.path.abspath(spec_dir), sh["texture"])
                if os.path.exists(src) and os.path.abspath(src) != os.path.join(out_dir, os.path.basename(src)):
                    shutil.copy(src, out_dir)
                sh["texture"] = "./" + os.path.basename(sh["texture"])
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{spec['name']}")
    stage.SetDefaultPrim(root.GetPrim())
    mode = spec.get("root") or {True: "rigid_links", False: "floating", None: "fixed"}[spec.get("fixed_base")]
    if mode not in ("fixed", "floating", "rigid_links"):
        raise ValueError(f"root must be fixed, floating or rigid_links, got {mode}")
    if mode != "rigid_links":
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    rp = f"/{spec['name']}"
    pm = UsdShade.Material.Define(stage, f"{rp}/physics_material")
    pmat = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    fr = spec.get("friction", [0.7, 0.6])
    pmat.CreateStaticFrictionAttr(fr[0]); pmat.CreateDynamicFrictionAttr(fr[1]); pmat.CreateRestitutionAttr(0.0)
    for link in spec["links"]:
        lp = f"{rp}/{link['name']}"
        lx = UsdGeom.Xform.Define(stage, lp)
        rb = UsdPhysics.RigidBodyAPI.Apply(lx.GetPrim())
        is_base = link is spec["links"][0]
        if is_base and mode == "rigid_links" and not spec.get("base_mass"):
            rb.CreateKinematicEnabledAttr(True)
        else:
            mass = spec.get("base_mass", link["mass"]) if is_base else link["mass"]
            UsdPhysics.MassAPI.Apply(lx.GetPrim()).CreateMassAttr(float(mass))
        for i, sh in enumerate(link["shapes"]):
            sp = f"{lp}/{sh.get('name', sh['type'] + str(i))}"
            t = sh["type"]
            if t == "cylinder":
                g = cylinder_mesh(stage, sp, sh["radius"], sh["height"])
                if sh.get("collision", True):
                    col = UsdGeom.Cylinder.Define(stage, sp + "_collision")
                    col.CreateRadiusAttr(sh["radius"]); col.CreateHeightAttr(sh["height"]); col.CreateAxisAttr("Z")
                    UsdGeom.Xformable(col).AddTranslateOp().Set(Gf.Vec3d(*sh["pos"]))
                    UsdGeom.Imageable(col).CreatePurposeAttr("guide")
                    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
                    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
            elif t == "box":
                g = UsdGeom.Cube.Define(stage, sp)
                g.CreateSizeAttr(1.0)
            elif t == "sphere":
                g = UsdGeom.Sphere.Define(stage, sp)
                g.CreateRadiusAttr(sh["radius"])
            elif t == "disc":
                g = disc_mesh(stage, sp, sh["radius"])
            elif t == "prism":
                g = prism_mesh(stage, sp, sh["polygon"], *sh["z"])
            else:
                raise ValueError(t)
            xf = UsdGeom.Xformable(g)
            xf.AddTranslateOp().Set(Gf.Vec3d(*sh.get("pos", [0, 0, 0])))
            if sh.get("yaw_deg"):
                xf.AddRotateZOp().Set(float(sh["yaw_deg"]))
            if t == "box":
                xf.AddScaleOp().Set(Gf.Vec3f(*sh["size"]))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material(stage, f"{sp}_look", sh, out_dir))
            if sh.get("visible", True) is False:
                UsdGeom.Imageable(g).CreatePurposeAttr("guide")
            if sh.get("collision", t != "disc") and t != "cylinder":
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                if t == "prism":
                    UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("convexHull")
                UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    links = {l["name"]: f"{rp}/{l['name']}" for l in spec["links"]}
    if mode == "fixed":
        fj = UsdPhysics.FixedJoint.Define(stage, f"{rp}/joints/root_joint")
        fj.CreateBody1Rel().SetTargets([links[spec["links"][0]["name"]]])
    for j in spec["joints"]:
        jp = f"{rp}/joints/{j['name']}"
        q = quat_from_x_to(j["axis"])
        a = Gf.Vec3f(*j["anchor"])
        if j["type"] == "revolute":
            jt = UsdPhysics.RevoluteJoint.Define(stage, jp)
            drive_type = "angular"
        else:
            jt = UsdPhysics.PrismaticJoint.Define(stage, jp)
            drive_type = "linear"
        jt.CreateAxisAttr("X")
        jt.CreateBody0Rel().SetTargets([links[j["parent"]]])
        jt.CreateBody1Rel().SetTargets([links[j["child"]]])
        jt.CreateLocalPos0Attr(a); jt.CreateLocalPos1Attr(a)
        jt.CreateLocalRot0Attr(q); jt.CreateLocalRot1Attr(q)
        jt.CreateLowerLimitAttr(float(j["limits"][0])); jt.CreateUpperLimitAttr(float(j["limits"][1]))
        drv = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), drive_type)
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(j.get("stiffness", 0.0)))
        drv.CreateDampingAttr(float(j.get("damping", 0.0)))
        drv.CreateTargetPositionAttr(float(j.get("target", 0.0)))
    for link_name, target in spec.get("filtered_pairs", []):
        fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(links[link_name]))
        fp.CreateFilteredPairsRel().AddTarget(f"{rp}/{target}")
    stage.GetRootLayer().Save()
    print(f"wrote {out}: links {list(links)} joints {[j['name'] for j in spec['joints']]} root {mode}")
    return out


if __name__ == "__main__":
    build(json.load(open(sys.argv[1])), os.path.dirname(os.path.abspath(sys.argv[1])))
