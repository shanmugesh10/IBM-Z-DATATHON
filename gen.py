import blenderproc as bproc
import numpy as np
import argparse, random, glob, os

# ------------ optional settings (you normally never need to touch these) ------------
HDRI_FOLDER = "hdris"
TARGET_SIZE = 0.2          # every object is scaled so its longest side is 0.2 m
PALETTE = [                # colors used for models that have no colors of their own
    [0.03, 0.03, 0.04, 1.0],   # black
    [0.60, 0.60, 0.62, 1.0],   # silver
    [0.10, 0.35, 0.20, 1.0],   # green
    [0.10, 0.20, 0.50, 1.0],   # blue
    [0.60, 0.05, 0.05, 1.0],   # red
    [0.85, 0.85, 0.85, 1.0],   # white
]
FIXED_COLOR = [0.6, 0.6, 0.62, 1.0]   # only used with --color fixed or jitter
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("model", help="path to the .obj file")
parser.add_argument("count", nargs="?", type=int, default=1000, help="images for this object")
parser.add_argument("--color", default="auto",
                    choices=["auto", "keep", "list", "random", "jitter", "fixed"],
                    help="auto = keep the model's own colors if it has any, else use the palette")
parser.add_argument("--distractors", action="store_true",
                    help="add random blocks that partly cover the object")
parser.add_argument("--res", type=int, default=640, help="image size in pixels")
parser.add_argument("--id", type=int, default=1,
                    help="ID number written into this object's mask pixels (1-255)")
args = parser.parse_intermixed_args()

MODEL_FILE = os.path.abspath(args.model)
OBJ_NAME = os.path.splitext(os.path.basename(MODEL_FILE))[0]
OUT_DIR = os.path.abspath(os.path.join("output", OBJ_NAME, "coco"))
IMG_DIR = os.path.join(OUT_DIR, "images")
MASK_DIR = os.path.abspath(os.path.join("output", OBJ_NAME, "masks"))
RAW_DIR = os.path.abspath(os.path.join("output", OBJ_NAME, "masks_raw"))
done = (len(glob.glob(os.path.join(IMG_DIR, "*.png"))) + len(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
        if os.path.isdir(IMG_DIR) else 0)
remaining = args.count - done


def model_has_own_colors(mats):
    """True if the model comes with textures or with several different colors."""
    colors = set()
    for m in mats:
        try:
            nt = m.blender_obj.node_tree
            if nt is not None:
                if any(n.type == "TEX_IMAGE" for n in nt.nodes):
                    return True
                bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]
                if bsdf and bsdf[0].inputs["Base Color"].is_linked:
                    return True
        except Exception:
            pass
        try:
            c = m.get_principled_shader_value("Base Color")
            colors.add(tuple(round(float(x), 2) for x in c[:3]))
        except Exception:
            pass
    return len(colors) >= 2


def save_png(path, arr):
    try:
        import cv2
        cv2.imwrite(path, arr)
        return
    except Exception:
        pass
    try:
        import imageio
        imageio.imwrite(path, arr)
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(path)


def main():
    bproc.init()
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. load the model; merge multi-part models into one object
    objs = bproc.loader.load_obj(MODEL_FILE)
    obj = objs[0]
    if len(objs) > 1:
        try:
            obj.join_with_other_objects(objs[1:])
            print("Merged", len(objs), "parts into one object")
        except Exception as e:
            print("WARNING: could not merge parts:", e)
    obj.set_cp("category_id", 1)

    # 2. center the geometry at the origin and scale it to TARGET_SIZE
    #    (so any model, tiny or huge, renders correctly without edits)
    try:
        bb = np.array(obj.get_bound_box())
        raw = float(np.max(bb.max(axis=0) - bb.min(axis=0)))
        obj.set_location(obj.get_location() - (bb.max(axis=0) + bb.min(axis=0)) / 2)
        obj.persist_transformation_into_mesh()
        obj.set_scale([TARGET_SIZE / raw] * 3)
        obj.persist_transformation_into_mesh()
    except Exception as e:
        print("WARNING: could not recenter/rescale object:", e)
    bb = np.array(obj.get_bound_box())
    size = float(np.max(bb.max(axis=0) - bb.min(axis=0)))
    print("Object:", OBJ_NAME, "| size after scaling:", round(size, 4))

    # 3. decide how to color the object
    obj_mats = obj.get_materials()
    mode = args.color
    if mode == "auto":
        mode = "keep" if model_has_own_colors(obj_mats) else "list"
        print("Color mode (auto):", mode,
              "- model has its own colors" if mode == "keep" else "- model has no colors, using palette")
    else:
        print("Color mode:", mode)

    if mode != "keep":
        mat = bproc.material.create("object_body")
        mat.set_principled_shader_value("Base Color", FIXED_COLOR)
        mat.set_principled_shader_value("Roughness", 0.35)
        mat.set_principled_shader_value("Metallic", 0.3)
        try:
            obj.replace_materials(mat)
        except Exception as e:
            print("replace_materials failed, using fallback:", e)
            if len(obj_mats) == 0:
                obj.add_material(mat)
        obj_mats = obj.get_materials()

    # 4. backgrounds
    hdri_dir = os.path.abspath(HDRI_FOLDER)
    hdris = glob.glob(os.path.join(hdri_dir, "*.hdr")) + glob.glob(os.path.join(hdri_dir, "*.exr"))
    assert len(hdris) > 0, "No HDRI files found in the hdris folder"

    # 5. render settings
    bproc.camera.set_resolution(args.res, args.res)
    bproc.renderer.set_max_amount_of_samples(64)
    bproc.renderer.set_noise_threshold(0.05)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    light = bproc.types.Light()
    light.set_type("POINT")

    distractors = []
    if args.distractors:
        for shape in ["CUBE", "SPHERE", "CYLINDER", "CONE", "MONKEY"] * 2:
            d = bproc.object.create_primitive(shape)
            d.set_cp("category_id", 2)
            dm = bproc.material.create("mat_" + str(len(distractors)))
            d.add_material(dm)
            distractors.append((d, dm))

    for i in range(remaining):
        bproc.utility.reset_keyframes()
        print("%s: image %d of %d" % (OBJ_NAME, done + i + 1, args.count))

        # random object position and rotation
        obj.set_location(np.random.uniform(-0.15 * size, 0.15 * size, 3))
        obj.set_rotation_euler(np.random.uniform(0, 2 * np.pi, 3))

        # random background, brightness and background rotation
        bproc.world.set_world_background_hdr_img(
            random.choice(hdris),
            strength=np.random.uniform(0.5, 2.0),
            rotation_euler=[0, 0, np.random.uniform(0, 2 * np.pi)])

        # random extra light
        light.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=2 * size,
                                               radius_max=6 * size,
                                               elevation_min=10, elevation_max=89))
        light.set_energy(np.random.uniform(20, 200) * (4 * size) ** 2)

        # random object color and finish
        if mode != "keep":
            if mode == "random":
                color = [float(c) for c in np.random.uniform(0.02, 0.9, 3)] + [1.0]
            elif mode == "list":
                color = random.choice(PALETTE)
            elif mode == "jitter":
                color = [float(np.clip(c * np.random.uniform(0.8, 1.2), 0, 1))
                         for c in FIXED_COLOR[:3]] + [1.0]
            else:
                color = FIXED_COLOR
            for m in obj_mats:
                m.set_principled_shader_value("Base Color", color)
                if mode in ("list", "random", "jitter"):
                    m.set_principled_shader_value("Roughness", float(np.random.uniform(0.15, 0.6)))
                    m.set_principled_shader_value("Metallic", float(np.random.uniform(0.0, 0.6)))

        # random distractors
        for d, dm in distractors:
            d.set_scale([np.random.uniform(0.2 * size, 0.5 * size)] * 3)
            d.set_location(np.random.uniform(-0.8 * size, 0.8 * size, 3))
            d.set_rotation_euler(np.random.uniform(0, 2 * np.pi, 3))
            dm.set_principled_shader_value("Base Color", [float(c) for c in np.random.uniform(0, 1, 3)] + [1])

        # random camera position, always looking near the object
        cam_loc = bproc.sampler.shell(center=[0, 0, 0], radius_min=1.8 * size,
                                      radius_max=5 * size,
                                      elevation_min=-20, elevation_max=85)
        poi = np.random.uniform(-0.1 * size, 0.1 * size, 3)
        rot = bproc.camera.rotation_from_forward_vec(
            poi - cam_loc, inplane_rot=np.random.uniform(-0.4, 0.4))
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc, rot))

        # render and save the image plus its bounding box
        data = bproc.renderer.render()
        before = set(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else set()
        bproc.writer.write_coco_annotations(
            OUT_DIR,
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="PNG",
            append_to_existing_output=True)

        # save the mask: masks/ = visible colored mask, masks_raw/ = numeric IDs (background 0, object = --id)
        try:
            segmap = np.array(data["instance_segmaps"][0])
            mask = np.zeros(segmap.shape[:2], dtype=np.uint8)
            for inst in data["instance_attribute_maps"][0]:
                if inst.get("category_id") == 1:
                    mask[segmap == inst["idx"]] = args.id
            new_files = [f for f in set(os.listdir(IMG_DIR)) - before if f.lower().endswith((".png", ".jpg"))]
            assert len(new_files) == 1, "could not tell which image was just written"
            stem = os.path.splitext(new_files[0])[0]
            save_png(os.path.join(RAW_DIR, stem + ".png"), mask)
            preview = np.zeros(mask.shape + (3,), dtype=np.uint8)
            preview[mask > 0] = [(37 * args.id) % 200 + 55, (91 * args.id) % 200 + 55,
                                 (173 * args.id) % 200 + 55]
            save_png(os.path.join(MASK_DIR, stem + ".png"), preview)
        except Exception as e:
            print("WARNING: could not save mask:", e)

    print("Finished %s. Images: %s | ID masks: %s" % (OBJ_NAME, IMG_DIR, MASK_DIR))


if remaining <= 0:
    print("%s already has %d images (target %d). Nothing to do." % (OBJ_NAME, done, args.count))
else:
    main()
