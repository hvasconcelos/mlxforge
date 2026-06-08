{
  "targets": [
    {
      "target_name": "mlxforge_node",
      "sources": ["src/addon.cc"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../src/capi"
      ],
      "defines": ["NAPI_CPP_EXCEPTIONS"],
      "cflags_cc": ["-std=c++17"],
      # Link the prebuilt libmlxforge.dylib from the repo's CMake build dir and
      # add an rpath to it so the addon finds it at runtime. A published package
      # would instead ship a prebuilt dylib alongside the .node file.
      "libraries": [
        "<(module_root_dir)/../../build/libmlxforge.dylib",
        "-Wl,-rpath,<(module_root_dir)/../../build"
      ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
        "MACOSX_DEPLOYMENT_TARGET": "13.0"
      }
    }
  ]
}
