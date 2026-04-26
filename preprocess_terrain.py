#!/usr/bin/env python3
"""Pre-process fixed2.tif terrain data into an ASC file for browser consumption.

The C++ simulator reads fixed2.tif via GDAL at 3x resolution (2718x1926),
then extracts slope/aspect/elevation by indexing every 3rd pixel:
    slope   = slopeTexTmp[3*cell + 0]
    aspect  = slopeTexTmp[3*cell + 1]
    elevation = slopeTexTmp[3*cell + 2]

This script reads the TIFF via PIL, resamples to 2718x1926, and writes
the raw float data as a flat binary file that the browser can fetch.

Output: static/data/terrain_sae.bin (slope/aspect/elevation interleaved, float32)
"""
import os
import struct
from PIL import Image

SIM_COLS = 906
SIM_ROWS = 642

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    tif_path = os.path.join(base, 'fire_sim_lib', 'data', 'fixed2.tif')
    out_path = os.path.join(base, 'static', 'data', 'terrain_sae.bin')

    img = Image.open(tif_path)
    # Resample to 3x grid dimensions (matching C++ GDAL RasterIO behavior)
    interp_w = SIM_COLS * 3
    interp_h = SIM_ROWS * 3
    img_resized = img.resize((interp_w, interp_h), Image.NEAREST)

    # Get pixel data as flat list
    pixels = list(img_resized.getdata())

    # Extract slope/aspect/elevation by indexing every 3rd pixel
    # cell = row * SIM_COLS + col; pixel index = 3 * cell + offset
    sim_size = SIM_COLS * SIM_ROWS
    sae = []
    for cell in range(sim_size):
        idx0 = 3 * cell
        idx1 = 3 * cell + 1
        idx2 = 3 * cell + 2
        if idx2 < len(pixels):
            # Handle multi-band or single-band pixels
            p0 = pixels[idx0] if isinstance(pixels[idx0], (int, float)) else pixels[idx0][0]
            p1 = pixels[idx1] if isinstance(pixels[idx1], (int, float)) else pixels[idx1][0]
            p2 = pixels[idx2] if isinstance(pixels[idx2], (int, float)) else pixels[idx2][0]
            sae.extend([float(p0), float(p1), float(p2)])
        else:
            sae.extend([0.0, 0.0, 0.0])

    # Write as float32 binary
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<%df' % len(sae), *sae))

    print('Wrote %d floats (%d cells x 3) to %s' % (len(sae), sim_size, out_path))
    print('File size: %.2f MB' % (os.path.getsize(out_path) / 1024 / 1024))

if __name__ == '__main__':
    main()
