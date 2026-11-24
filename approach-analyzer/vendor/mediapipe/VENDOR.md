# Vendored MediaPipe files

The Approach Analyzer loads these from this repo so the page makes no
third-party requests for body tracking.

| File | Source | Version |
| --- | --- | --- |
| `vision_bundle.mjs` | npm `@mediapipe/tasks-vision` | 1.2.3 |
| `wasm/vision_wasm_internal.js` | npm `@mediapipe/tasks-vision` | 1.2.3 |
| `wasm/vision_wasm_internal.wasm` | npm `@mediapipe/tasks-vision` | 1.2.3 |
| `pose_landmarker_full.task` | Google's model link from the [Pose Landmarker docs](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) | SHA-256 `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad` |

The library and model are licensed under Apache 2.0.

## Updating

- Library: run `npm pack @mediapipe/tasks-vision@<version>`, then copy
  `vision_bundle.mjs` and the two `wasm/vision_wasm_internal.*` files here.
- Model: run the **Fetch pose model** GitHub Actions workflow
  (`.github/workflows/fetch-pose-model.yml`). It downloads the file, checks
  that it's a real model bundle, and commits it if it changed.

## Privacy note

`vision_bundle.mjs` sends usage logs to `https://odml.pa.googleapis.com/v1/log`
every 60 seconds, and the library has no option to turn that off. The files are
left unmodified. Instead, `approach-analyzer/index.html` sets a
Content-Security-Policy of `connect-src 'self' blob: data:`, so the browser
blocks the call. You'll see a CSP error for that URL in the browser console.
That error is expected and means the block is working. Keep that meta tag if
you edit the page, and check the console for the CSP error after any library
update.
