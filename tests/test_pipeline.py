"""Algorithm and CLI regression tests; no image-processing library calls."""
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from PIL import Image
from vision import (grayscale,median3,gaussian3,intermeans,segment,trace_boundary,
                    resample,chain_code,code_distance,cluster)


def polygon():
    return [(math.cos(i*math.tau/32)*(1+.22*math.cos(3*i*math.tau/32)),
             .7*math.sin(i*math.tau/32)*(1+.22*math.cos(3*i*math.tau/32))) for i in range(32)]


class Algorithms(unittest.TestCase):
    def test_grayscale(self):
        self.assertEqual(grayscale([(255,0,0)]*4,2,2),[[54,54],[54,54]])
    def test_median_impulse(self):
        a=[[30]*7 for _ in range(7)]; a[3][3]=255
        self.assertEqual(median3(a)[3][3],30)
    def test_gaussian_constant(self):
        self.assertEqual(gaussian3([[40]*5 for _ in range(4)]),[[40]*5 for _ in range(4)])
    def test_gaussian_reduces_variance(self):
        rng=random.Random(7); a=[[round(rng.gauss(120,10)) for _ in range(40)] for _ in range(40)]
        mse=lambda im:sum((v-120)**2 for row in im for v in row)/1600
        self.assertLess(mse(gaussian3(a)),mse(a))
    def test_segmentation_polarity_holes_noise(self):
        for bg,fg in [(20,200),(230,30)]:
            a=[[bg]*40 for _ in range(40)]
            for y in range(10,30):
                for x in range(8,32): a[y][x]=fg
            a[15][15]=bg; a[3][3]=fg
            mask,_=segment(a)
            self.assertEqual(sum(map(sum,mask)),480)
            self.assertEqual(len(resample(trace_boundary(mask))),32)
    def test_blank_rejected(self):
        with self.assertRaises(ValueError): intermeans([[5]*4 for _ in range(4)])
    def test_exact_polygon_rst_and_start(self):
        p=polygon(); original=chain_code(p)
        for angle in [0,.123,.71,1.73,math.pi]:
            transformed=[(2.4*(x*math.cos(angle)-y*math.sin(angle))+17,
                          2.4*(x*math.sin(angle)+y*math.cos(angle))-9) for x,y in p]
            for k in range(32):
                self.assertEqual(chain_code(transformed[k:]+transformed[:k]),original)
    def test_distance(self):
        a=chain_code(polygon());b=a[9:]+a[:9]
        self.assertAlmostEqual(code_distance(a,b),0,places=7)
        self.assertAlmostEqual(code_distance(a,a),0,places=7)
        with self.assertRaises(ValueError): code_distance('1',a)
    def test_distinct_shapes(self):
        circle=[(math.cos(i*math.tau/32),math.sin(i*math.tau/32)) for i in range(32)]
        slender=[(x*3,y*.4) for x,y in circle]
        self.assertGreater(code_distance(chain_code(circle),chain_code(slender)),.16)
    def test_complete_link(self):
        a=chain_code(polygon())
        groups,_=cluster([a,a],.001); self.assertEqual(groups,[[0,1]])
    def test_raster_rst(self):
        codes=[]
        for angle,scale,tx,ty in [(0,1,0,0),(.7,.75,6,-7),(1.9,1.1,-4,3)]:
            a=[]
            for y in range(120):
                row=[]
                for x in range(140):
                    u=(x-70-tx)/scale;v=(y-60-ty)/scale
                    rx=u*math.cos(angle)+v*math.sin(angle)
                    ry=-u*math.sin(angle)+v*math.cos(angle)
                    row.append(210 if (rx/35)**2+(ry/20)**2<1 else 20)
                a.append(row)
            mask,_=segment(gaussian3(median3(a)))
            codes.append(chain_code(resample(trace_boundary(mask))))
        self.assertEqual(len(cluster(codes,.16)[0]),1)


class CommandLine(unittest.TestCase):
    def test_formats_corruption_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inputs=root/'inputs';inputs.mkdir()
            rgb=[(200,200,200) if 10<x<40 and 10<y<40 else (20,20,20) for y in range(50) for x in range(50)]
            im=Image.new('RGB',(50,50));im.putdata(rgb)
            for ext in ['PNG','jpg','jpeg']: im.save(inputs/f'test.{ext}')
            output=root/'out'/'groups.txt'
            cmd=[sys.executable,'main.py',str(inputs),'--output',str(output),'--diagnostics',str(root/'diag')]
            run=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('Processed: 3; failed: 0',output.read_text())
            (inputs/'broken.png').write_bytes(b'bad')
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,1)
            self.assertIn('broken.png',output.read_text())
            empty=root/'empty';empty.mkdir()
            self.assertEqual(subprocess.run([sys.executable,'main.py',str(empty)],capture_output=True).returncode,2)


if __name__=='__main__': unittest.main()
