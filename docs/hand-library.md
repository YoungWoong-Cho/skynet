# Hand library

The Hands page stores pinned robot descriptions, licenses and real mesh assets. It supports joint limits, mimic joints, orbit/zoom/pan, and local poses scoped to a model, side and source revision. Exported poses use radians (meters for prismatic joints). This is a geometry/kinematics viewer, not physics, collision checking, retargeting or hardware control.

`config/hands.json` lists sources and explicit compatibility exceptions. Twelve variants across all seven requested models have been installed and checked locally. Allegro V4 left is explicitly unavailable because its official URDF references the absent `link_12.0_left.STL`. LEAP V1 only supplies the right-hand model. Inspire is labeled third-party Renesas. Shadow uses official E3M5/PST descriptions; this joint layout differs from DexVerse's floating Shadow hand. Do not transfer poses between those layouts without a mapping.

Descriptions and assets are downloaded once into `data/hands/<model>/<revision>/<side>`. Downloads are bounded and published atomically only after all referenced assets are stored. Git LFS assets are checked against the source pointer's SHA-256 and byte count. The manifest records every asset checksum; serving an existing model needs no GitHub or GPU access. Mesh metadata is cached in memory by manifest modification time. A failed or unsupported model shows an explicit error instead of substitute geometry.

Model licenses remain with the local assets and are linked from the page. Original upstream assets are not added to this Git repository. Generated Shadow URDFs retain their copyright/license notices. Reproduce them from the catalog's pinned official source using:

```sh
uv run --no-project --with xacro==2.1.1 python ops/hands/generate_shadow.py
```

The browser supports visual STL, COLLADA and GLB files. Original collision meshes are retained, but not displayed. WUJI Hand 2's preview rotation keeps the hand upright; this affects only the displayed root orientation and does not alter URDF coordinates or saved joint values. Rendering happens on interaction or resize rather than continuously. Model downloads and loading errors are bounded; switching models disposes GPU geometry and materials.

To rebuild the committed browser bundle:

```sh
npm ci
npm run build:hands
npm run test:hands
.venv/bin/python -m pytest tests/test_hands.py
```

Add future hands by pinning their repository revision, supplied sides, license path, and ROS package mapping in the catalog. Unsupported descriptions need an explicit converter/loader before being enabled. DexVerse integration additionally requires compatible robot assets, task configuration and retargeting/control mappings; adding a URDF here does not assert those are available.
