# Native machine vision pipeline

A Python implementation of the supplied machine-vision assignment. It loads PNG,
JPG and JPEG images, suppresses noise, thresholds a single object, extracts its
boundary, creates an invariant **32-digit octal chain code**, and groups similar
outlines. All processing and matching algorithms are implemented in `vision.py`
using Python lists, loops and the standard library. Pillow is used only for image
file decoding, RGB mode decoding and saving diagnostic images. No NumPy, OpenCV,
SciPy, scikit-image or built-in image filters are used.

## Install and run on Windows

Requires Python 3.10 or newer. From PowerShell:

```powershell
cd D:\mV\machine-vision-pipeline
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py "..\Samples (1)"
```

For another folder:

```powershell
.\.venv\Scripts\python.exe main.py "C:\path\to\images" --output results\groups.txt
```

Linux/macOS equivalents: `python3 -m venv .venv`,
`.venv/bin/python -m pip install -r requirements.txt`, then
`.venv/bin/python main.py /path/to/images`.

## Check the results

Open `results/groups.txt` for grouped filenames. `results/groups.json` contains
all codes, sampled points, thresholds, dimensions, pairwise distances, timing,
parameters and failures. `results/diagnostics` contains grayscale, filtered,
filled mask and boundary sample images for each input. Gray dots in each
`*_boundary.png` mark the 32 samples on the white silhouette.

Run verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The 12 tests cover filtering, segmentation polarity, holes, isolated noise,
constant-image rejection, exact continuous polygon RST/start invariance,
raster transformation tolerance, distinct shapes, clustering, and CLI image
formats/error handling. Raster tolerance is tested at a specified 0.16 cutoff;
it is not a guarantee that all transformed images match at every cutoff.

The supplied 12 samples were run with the default settings and zero input
failures. The saved result has three groups:

- `im-1.png` through `im-6.png`
- `im-10.png`, `im-12.png`, `im-9.png`
- `im-11.png`, `im-7.png`, `im-8.png`

These are unsupervised outline-similarity groups, not verified object identities.
In particular, `im-7.png` looks like the textured rock family but joins a different
outline group. This is a concrete limitation of the coarse code and this cutoff;
there are no supplied ground-truth labels from which to claim accuracy. Changing
the threshold can merge or split groups. No filenames or expected memberships
are encoded in the algorithm.

## Processing and invariance

1. Decode RGB and calculate luminance `0.21 R + 0.72 G + 0.07 B`.
2. Reduce large images to at most 320 pixels on the longest side using native
   box averaging; preserve aspect ratio. This bounds pure-Python processing cost.
3. Apply a native 3 by 3 median filter, then separable `[1,2,1]/4` Gaussian
   approximation, with replicated borders.
4. Estimate the threshold using lecture Unit 6 inter-means. Infer foreground
   polarity from the frame border. A robust border estimate (trimmed mean plus
   or minus three standard deviations, at least eight levels) adjusts the
   threshold to retain dark shadows and reject background noise.
5. Keep the largest 4-connected component, pad it, and fill enclosed holes by
   exterior flood fill. This assumes one dominant, solid object and a border
   that mostly shows background.
6. Follow the outer boundary with Moore neighbor tracing and a directed-edge
   stopping rule. Smooth coordinates with a circular moving average, radius
   `max(1, boundary_pixel_count // 128)`, then interpolate 32 equal Euclidean
   arc-length samples (Unit 7 method 3).
7. For each cyclic origin, subtract the first segment's continuous heading
   from all headings, then quantize into eight directions. Select the
   lexicographically smallest 32-digit sequence. These are relative direction
   codes, not differential turn codes. Translation disappears in differences;
   perimeter fractions remove uniform scale; heading subtraction removes
   rotation before quantization; cyclic minimization removes sample origin.
8. Decode only the codes into unit-step polygons. Spread closure error evenly,
   center and RMS-normalize, and find the best cyclic correspondence and
   analytic planar rotation. Distance is half the normalized RMS residual.
9. Merge the closest clusters only when **every** cross-cluster pair is within
   the cutoff (complete linkage). Save groups in deterministic input order.

Exact invariance holds for continuous sampled polygons under a positive uniform
scale, rigid rotation, translation and cyclic reindexing, away from numerical
quantization ties. Resegmentation, raster interpolation and a changed contour
sampling phase introduce approximation. No finite 32-digit descriptor uniquely
identifies every possible shape. Mirror invariance, affine/perspective changes,
multiple objects, occlusion, texture recognition and hole topology are outside
scope. There is no universal error-free classification guarantee.

## CLI options and failure behavior

```text
main.py INPUT [--output PATH] [--diagnostics DIR]
              [--threshold FLOAT] [--max-size INTEGER]
```

- `INPUT`: nonrecursive folder; suffix matching is case-insensitive.
- `--output`: default `results/groups.txt`; adjacent JSON is also written.
- `--diagnostics`: default `results/diagnostics`.
- `--threshold`: default `0.06`; range 0 to 1. Smaller is stricter. This is an
  exploratory sample setting, not independently trained or validated.
- `--max-size`: default `320`; minimum `32`. Higher costs more time and memory.
- Exit 0: every selected image processed. Exit 1: one or more selected images
  failed (valid results and failures are still saved). Exit 2: invalid arguments,
  missing directory or no supported images. Unexpected I/O write failures are
  surfaced rather than reported as successful.

Do not use the input directory as the diagnostics directory: generated PNGs
would become inputs on the next run. Output files overwrite prior results;
old diagnostic files from previous, different inputs should be cleared manually.
Transparent inputs use their decoded RGB values; use opaque inputs for
predictable segmentation. Objects touching the frame can be traced using padding,
but can make the border-based background assumption unreliable.

## Repository layout

- `main.py`: CLI, image I/O, text/JSON and diagnostic outputs.
- `vision.py`: native algorithms and chain-code-only clustering.
- `tests/test_pipeline.py`: deterministic regression tests.
- `requirements.txt`: image I/O dependency.
- `results/`: actual supplied-sample outputs and diagnostics.
- `report.docx`: completed brief technical report with pseudocode and results.
- `tools/package_submission.py`: reproducible submission ZIP builder.

The supplied images and lecture PDFs stay in their original sibling folders;
they are not required to execute on a different input folder. No external model,
dataset, internet access or filename-specific rule is needed.

## Submission archive

```powershell
.\.venv\Scripts\python.exe tools\package_submission.py --index 220155B
```

Creates `submission/220155B.zip` containing source, README, requirements, tests,
report and actual text/JSON results. The assignment requests source, report and
output text; diagnostic images are retained in the repository for inspection.
ZIPs and virtual environments are excluded from Git. Extract the ZIP, install
requirements and pass an image folder to `main.py` to reproduce a run.

## Lecture basis

The implementation follows the supplied UOM Computer Vision / Machine Vision
lectures: Unit 3a for luminance and normalization; Unit 4 for median/Gaussian
filtering; Image as Functions for convolution and border handling; Unit 6 for
inter-means thresholding, connected regions and contour tracking; Unit 7 for
boundary chain codes, equal-length sampling and normalization. Units 1, 2 and
input geometry motivate the assumptions about illumination, sampling and
geometric variation. The border refinement, continuous-heading normalization,
code reconstruction distance and complete-link clustering are implementation
choices extending those concepts, not claims that those exact algorithms were
specified in the lectures.
