# Inside PointPillars

A clickable 3D walkthrough of one LiDAR scan moving through the PointPillars object detector, from raw points to final boxes. It runs entirely in the browser.

**Live:** https://tbfouts.github.io/inside-pointpillars/

## The eight stages

| # | Stage | Where it runs in a TensorRT deployment |
|---|---|---|
| 1 | Points | Input, copied to the GPU |
| 2 | Pillars (voxelization) | CUDA kernel, outside the engine |
| 3 | Pillar features | TensorRT layers |
| 4 | Scatter to the bird's-eye-view grid | TensorRT plugin (hand-written CUDA inside the engine) |
| 5 | 2D backbone | TensorRT layers |
| 6 | Detection head | TensorRT layers |
| 7 | Box decoding | CUDA kernel, outside the engine |
| 8 | Non-maximum suppression | CUDA kernels outside the engine, finished by a CPU loop |

Each stage shows the tensors going in and out, labeled as **features** (different every scan) or **weights** (fixed after training).

## What's real and what's illustrative

- **Real:** the grid (432 × 496 cells of 0.16 m), the 40,000-pillar padding, up to 32 points per pillar, every tensor shape, the anchors and the score/NMS thresholds. These follow the KITTI configuration used by NVIDIA's [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars), checked against its source at commit `ce7e2bd` (December 2023).
- **Generated:** the street and the LiDAR scan are ray-cast in your browser from a simple synthetic scene (a 64-beam sensor, 0.25° azimuth steps).
- **Illustrative:** the pillar-feature weights are hand-set or random, and the detection scores are simulated around the scene's real objects. The arithmetic in each stage is real; the numbers are not a trained model's output.

## Running locally

It's a single static file. Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
```

It loads three.js (r128) from cdnjs and the Schibsted Grotesk and Spline Sans Mono fonts from Google Fonts, so it needs internet access.

## Notes

Personal learning project. Not affiliated with or endorsed by NVIDIA.
