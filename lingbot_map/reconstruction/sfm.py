"""Independent CPU feature tracks and bundle-adjusted cameras through COLMAP 4."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from .io import digest, write_json


def reconstruct_cameras(output):
    output=Path(output)
    executable=shutil.which("colmap")
    if executable is None:
        raise ValueError("Install the Mac photogrammetry backend with: brew install colmap")
    root=output/"colmap"
    root.mkdir(exist_ok=True)
    database=root/"database.db"
    sparse=root/"sparse"
    snapshots=root/"snapshots"
    sparse.mkdir(exist_ok=True);snapshots.mkdir(exist_ok=True)
    signature={"input_sha256":digest(output/"input.json"),"camera":"PINHOLE",
               "features":4096,"image_size":1280,"matching_overlap":10,"schema":1}
    config=root/"configuration.json"
    if config.exists() and json.loads(config.read_text())!=signature:
        raise ValueError("Photogrammetry configuration changed; use a new output directory")
    if not config.exists():write_json(config,signature)
    commands=[("features",["feature_extractor","--database_path",str(database),"--image_path",str(output/"frames"),
                "--ImageReader.single_camera","1","--ImageReader.camera_model","PINHOLE",
                "--FeatureExtraction.use_gpu","0","--FeatureExtraction.num_threads","6",
                "--FeatureExtraction.max_image_size","1280","--SiftExtraction.max_num_features","4096"]),
              ("matching",["sequential_matcher","--database_path",str(database),
                "--FeatureMatching.use_gpu","0","--FeatureMatching.num_threads","6",
                "--SequentialMatching.overlap","10","--SequentialMatching.quadratic_overlap","1"]),
              ("mapping",["mapper","--database_path",str(database),"--image_path",str(output/"frames"),
                "--output_path",str(sparse),"--Mapper.num_threads","6",
                "--Mapper.snapshot_path",str(snapshots),"--Mapper.snapshot_frames_freq","50"])]
    for name,command in commands:
        marker=root/(name+".complete.json")
        if marker.exists():continue
        print("Photogrammetry: "+name,file=sys.stderr,flush=True)
        with (root/(name+".log")).open("a") as log:
            result=subprocess.run([executable]+command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Photogrammetry {name} failed; inspect {root/(name+'.log')}")
        write_json(marker,{"command":command,"returncode":result.returncode})
    models=[]
    for model in sorted(sparse.iterdir()):
        if not (model/"cameras.bin").exists():continue
        text=model/"text"
        text.mkdir(exist_ok=True)
        if not (text/"images.txt").exists():
            subprocess.run([executable,"model_converter","--input_path",str(model),
                            "--output_path",str(text),"--output_type","TXT"],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        models.append(str(text.relative_to(output)))
    with sqlite3.connect(database) as db:
        counts={"images":db.execute("select count(*) from images").fetchone()[0],
                "verified_pairs":db.execute("select count(*) from two_view_geometries where rows>0").fetchone()[0]}
    write_json(root/"result.json",{"models":models,**counts,
               "units":"unknown scale", "loop_detection":"temporal and quadratic pairs only"})
    if not models:
        raise ValueError("Photogrammetry could not register a camera model; inspect mapping.log")
