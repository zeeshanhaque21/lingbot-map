"""Keep captured appearance from being illuminated a second time by the viewer."""


def unlit_materials(tree):
    extension = "KHR_materials_unlit"
    used = tree.setdefault("extensionsUsed", [])
    if extension not in used:
        used.append(extension)
    materials = tree.setdefault("materials", [])
    if not materials:
        materials.append(
            {"pbrMetallicRoughness": {"metallicFactor": 0, "roughnessFactor": 1}}
        )
    for material in materials:
        material.setdefault("extensions", {})[extension] = {}
        material["doubleSided"] = True
    for mesh in tree["meshes"]:
        for primitive in mesh["primitives"]:
            primitive.setdefault("material", 0)
