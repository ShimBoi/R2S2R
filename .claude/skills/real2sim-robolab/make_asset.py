"""Turn a mesh into a RoboLab rigid object USD with measured physical properties.

Input: a USD from the RoboLab/Objaverse library (textures are copied along), or an .obj/.glb/.gltf/.ply/.stl
(procedural or image-to-3D output) loaded with trimesh. Output: <out_dir>/<name>.usd(a) whose default
prim is the rigid body, origin at the bottom centre of the bounding box (so presets can use z = support surface + 0.005).

Usage:
  python make_asset.py SRC OUT_DIR --name obj_a --mass 0.04 [--longest-m 0.145 | --scale 0.9] [--rot-deg 0 90 0]
      [--static-friction 1.2 --dynamic-friction 1.0 --restitution 0.0]
      [--collision convexHull|convexDecomposition|boundingCube|boundingSphere|sdf|none]
      [--com x y z] [--compliant-stiffness 2000 --compliant-damping 20] [--texture-dir DIR ...] [--color r g b]
"""
import argparse
import os
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial import ConvexHull


def strip_physics(stage, root):
    for p in list(root.GetChildren()):
        if p.GetName() in ("GroundPlane", "Render", "physicsScene", "PhysicsScene"):
            stage.RemovePrim(p.GetPath())
    for p in stage.Traverse():
        if p == root:
            continue
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            for n in ("physics:rigidBodyEnabled", "physics:kinematicEnabled"):
                if p.HasAttribute(n):
                    p.RemoveProperty(n)
        if p.HasAPI(UsdPhysics.MassAPI):
            p.RemoveAPI(UsdPhysics.MassAPI)
            for n in [x.GetName() for x in p.GetAttributes() if x.GetName().startswith(("physics:mass", "physics:density",
                      "physics:centerOf", "physics:diagonal", "physics:principal"))]:
                p.RemoveProperty(n)


def from_usd(src, out, texture_dirs):
    stage = Usd.Stage.Open(Usd.Stage.Open(src).Flatten())
    root = stage.GetDefaultPrim()
    strip_physics(stage, root)
    for td in texture_dirs:
        shutil.copytree(td, os.path.join(os.path.dirname(out), os.path.basename(td)), dirs_exist_ok=True)
    for p in stage.Traverse():
        for at in p.GetAttributes():
            if at.GetTypeName() == Sdf.ValueTypeNames.Asset and at.Get():
                path = at.Get().path
                for td in texture_dirs:
                    if path.startswith(td + "/"):
                        at.Set(Sdf.AssetPath("./" + os.path.basename(td) + path[len(td):]))
    stage.GetRootLayer().Export(out)
    return Usd.Stage.Open(out)


def from_trimesh(src, out, name, color):
    import trimesh
    from PIL import Image

    m = trimesh.load(src, force="mesh")
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in m.vertices]))
    mesh.CreateFaceVertexCountsAttr([3] * len(m.faces))
    mesh.CreateFaceVertexIndicesAttr(m.faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mat = UsdShade.Material.Define(stage, f"/{name}/Looks/Surface")
    sh = UsdShade.Shader.Define(stage, f"/{name}/Looks/Surface/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    vis = m.visual
    img = getattr(getattr(vis, "material", None), "baseColorTexture", None) or getattr(getattr(vis, "material", None), "image", None)
    if getattr(vis, "uv", None) is not None and img is not None:
        tex = f"{name}_albedo.png"
        Image.fromarray(np.asarray(img.convert("RGB"))).save(os.path.join(os.path.dirname(out), tex))
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, uv)) for uv in vis.uv]))
        rd = UsdShade.Shader.Define(stage, f"/{name}/Looks/Surface/Reader")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tx = UsdShade.Shader.Define(stage, f"/{name}/Looks/Surface/Texture")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./" + tex))
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    elif getattr(vis, "kind", None) == "vertex" and color is None:
        c = np.asarray(vis.vertex_colors)[:, :3] / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, x)) for x in lin]))
        rd = UsdShade.Shader.Define(stage, f"/{name}/Looks/Surface/ColorReader")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    else:
        c = np.array(color or (0.7, 0.7, 0.7)) / (255.0 if max(color or (0,)) > 1 else 1.0)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, c)))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--longest-m", type=float)
    ap.add_argument("--static-friction", type=float, default=0.8)
    ap.add_argument("--dynamic-friction", type=float, default=0.6)
    ap.add_argument("--restitution", type=float, default=0.0)
    ap.add_argument("--collision", default="convexHull")
    ap.add_argument("--rot-deg", type=float, nargs=3, help="rotate about x, y, z before scaling (lay the object down)")
    ap.add_argument("--com", type=float, nargs=3)
    ap.add_argument("--compliant-stiffness", type=float)
    ap.add_argument("--compliant-damping", type=float, default=0.0)
    ap.add_argument("--texture-dir", nargs="*", default=[])
    ap.add_argument("--color", type=float, nargs=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    is_usd = a.src.endswith((".usd", ".usda", ".usdc"))
    out = os.path.join(a.out_dir, f"{a.name}.usd" if is_usd else f"{a.name}.usda")
    stage = from_usd(a.src, out, a.texture_dir) if is_usd else from_trimesh(a.src, out, a.name, a.color)
    root = stage.GetDefaultPrim()

    kids = [c for c in root.GetChildren() if c.IsA(UsdGeom.Xformable)]

    def bake(extra):
        for k in kids:
            kx = UsdGeom.Xformable(k)
            m = kx.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            kx.ClearXformOpOrder()
            kx.AddTransformOp().Set(m * extra)
        UsdGeom.Xformable(root).ClearXformOpOrder()

    if a.rot_deg:
        r = Gf.Rotation(Gf.Vec3d(1, 0, 0), a.rot_deg[0]) * Gf.Rotation(Gf.Vec3d(0, 1, 0), a.rot_deg[1]) * Gf.Rotation(Gf.Vec3d(0, 0, 1), a.rot_deg[2])
        bake(Gf.Matrix4d().SetRotate(r))
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root).ComputeAlignedRange()
    s = a.scale or (a.longest_m / max(bb.GetSize()) if a.longest_m else 1.0)
    shift = -Gf.Vec3d((bb.GetMin()[0] + bb.GetMax()[0]) / 2, (bb.GetMin()[1] + bb.GetMax()[1]) / 2, bb.GetMin()[2])
    bake(Gf.Matrix4d().SetTranslate(shift) * Gf.Matrix4d().SetScale(s))

    UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(a.mass)
    if a.com:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*a.com))
    pm = UsdShade.Material.Define(stage, root.GetPath().AppendPath("PhysicsMaterial"))
    pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    pma.CreateStaticFrictionAttr(a.static_friction)
    pma.CreateDynamicFrictionAttr(a.dynamic_friction)
    pma.CreateRestitutionAttr(a.restitution)
    if a.compliant_stiffness:
        pm.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
        pm.GetPrim().CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(a.compliant_stiffness)
        pm.GetPrim().CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(a.compliant_damping)
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    colliders = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI)] or meshes
    pts = []
    for p in meshes:
        xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pts += [list(xf.Transform(Gf.Vec3d(*v))) for v in (UsdGeom.Mesh(p).GetPointsAttr().Get() or [])]
    for p in colliders:
        if a.collision == "none":
            continue
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr(a.collision)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()

    chk = Usd.Stage.Open(out)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(chk.GetDefaultPrim()).ComputeAlignedRange()
    vol = ConvexHull(np.array(pts)).volume if len(pts) > 3 else float("nan")
    print(f"wrote {out}\n  size {[round(v, 4) for v in r.GetSize()]} m, min {[round(v, 4) for v in r.GetMin()]}, scale {s:.4f}")
    print(f"  mass {a.mass} kg, hull volume {vol * 1e6:.1f} cm^3, mass / hull volume {a.mass / vol:.0f} kg/m^3"
          f"\n  friction s/d {a.static_friction}/{a.dynamic_friction}, restitution {a.restitution}, collision {a.collision}, "
          f"{len(colliders)} collider mesh(es), compliant {a.compliant_stiffness}")


if __name__ == "__main__":
    main()
