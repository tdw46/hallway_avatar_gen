# Quad Remesher Runtime Layout

Hallway Avatar Gen resolves the Quad Remesher bridge runtime by platform.

Expected folders:

- `Darwin-arm64/`
- `Darwin-x64/`
- `Windows-x64/`
- `Windows-arm64/`
- `Linux-x64/`
- `Linux-arm64/`

Each platform folder should contain the native `qmesh` executable for that
platform plus its companion runtime files and `resources/` directory.

Examples:

- macOS: `qmesh`, `qmeshlib.dylib`, `libfbxsdk.dylib`, `ChSolver.dylib`, `resources/`
- Windows: `qmesh.exe`, `qmeshlib.dll`, required `.dll` files, `resources/`
- Linux: `qmesh`, `libqmeshlib.so`, required `.so` files, `resources/`

The legacy macOS layout directly in `resources/quad_remesher_engine/` is still
accepted for backward compatibility, but cross-platform builds should use the
platform subfolders above.
