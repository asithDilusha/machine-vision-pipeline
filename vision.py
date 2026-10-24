"""Native image processing and invariant 32-direction boundary descriptors.

Pillow is deliberately restricted to decoding/encoding in the CLI module.
Coordinates use x right, y down; contours have clockwise orientation.
"""
from collections import deque
from math import atan2, floor, hypot, pi, cos, sin, sqrt


def grayscale(rgb, width, height, max_size=320):
    """Luminance and native box reduction, preserving the aspect ratio."""
    scale = min(1.0, max_size / max(width, height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            values = []
            for sy in range(y * height // h, (y + 1) * height // h):
                for sx in range(x * width // w, (x + 1) * width // w):
                    r, g, b = rgb[sy * width + sx]
                    values.append(.21 * r + .72 * g + .07 * b)
            row.append(round(sum(values) / len(values)))
        out.append(row)
    return out


def median3(image):
    """3 by 3 order statistic filter with replicated image borders."""
    h, w = len(image), len(image[0])
    return [[sorted(image[min(h-1,max(0,y+dy))][min(w-1,max(0,x+dx))]
                    for dy in (-1,0,1) for dx in (-1,0,1))[4]
             for x in range(w)] for y in range(h)]


def gaussian3(image):
    """Separable binomial Gaussian approximation [1,2,1]/4."""
    h, w = len(image), len(image[0])
    horizontal = [[(row[max(0,x-1)] + 2*row[x] + row[min(w-1,x+1)])/4
                   for x in range(w)] for row in image]
    return [[round((horizontal[max(0,y-1)][x] + 2*horizontal[y][x]
                    + horizontal[min(h-1,y+1)][x])/4)
             for x in range(w)] for y in range(h)]


def intermeans(image):
    """Lecture Unit 6 iterative histogram threshold; bounded convergence."""
    hist = [0]*256
    for row in image:
        for value in row:
            hist[value] += 1
    if sum(v > 0 for v in hist) < 2:
        raise ValueError('Uniform image has no separable foreground')
    threshold = sum(i*n for i,n in enumerate(hist))/sum(hist)
    for _ in range(100):
        cut = int(threshold)
        lo, hi = sum(hist[:cut+1]), sum(hist[cut+1:])
        if not lo or not hi:
            break
        updated = (sum(i*hist[i] for i in range(cut+1))/lo +
                   sum(i*hist[i] for i in range(cut+1,256))/hi)/2
        if abs(updated-threshold) < .05:
            threshold = updated
            break
        threshold = updated
    return threshold


def segment(image):
    """Threshold, infer polarity from border, retain largest 4-component, fill holes."""
    h, w = len(image), len(image[0])
    threshold = intermeans(image)
    border = image[0] + image[-1] + [row[0] for row in image[1:-1]] + [row[-1] for row in image[1:-1]]
    bright_background = sum(v > threshold for v in border) > len(border)/2
    # Robust border statistics preserve shadowed foreground missed by a midpoint.
    ordered = sorted(border)
    central = ordered[len(ordered)//10:len(ordered)*9//10]
    mean = sum(central)/len(central)
    sigma = (sum((v-mean)**2 for v in central)/len(central))**.5
    if bright_background:
        threshold = max(threshold, mean-max(8,3*sigma))
    else:
        threshold = min(threshold, mean+max(8,3*sigma))
    mask = [[(v <= threshold if bright_background else v > threshold) for v in row] for row in image]
    seen, largest = set(), []
    for y in range(h):
        for x in range(w):
            if not mask[y][x] or (x,y) in seen:
                continue
            component, queue = [], deque([(x,y)])
            seen.add((x,y))
            while queue:
                p = queue.popleft(); component.append(p)
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny = p[0]+dx,p[1]+dy
                    if 0<=nx<w and 0<=ny<h and mask[ny][nx] and (nx,ny) not in seen:
                        seen.add((nx,ny)); queue.append((nx,ny))
            if len(component)>len(largest):
                largest = component
    if len(largest)<16:
        raise ValueError('No usable object component (fewer than 16 pixels)')
    # Padding makes exterior flood fill and contour tracing safe at the frame edge.
    result = [[False]*(w+2) for _ in range(h+2)]
    for x,y in largest:
        result[y+1][x+1] = True
    outside = {(0,0)}; queue = deque([(0,0)])
    while queue:
        x,y = queue.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<w+2 and 0<=ny<h+2 and not result[ny][nx] and (nx,ny) not in outside:
                outside.add((nx,ny)); queue.append((nx,ny))
    for y in range(1,h+1):
        for x in range(1,w+1):
            result[y][x] = (x,y) not in outside
    return result, threshold


DIRECTIONS = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))


def trace_boundary(mask):
    """Moore neighbor tracing with directed-edge stopping criterion."""
    start = next(((x,y) for y,row in enumerate(mask) for x,v in enumerate(row) if v), None)
    if start is None:
        raise ValueError('Empty mask')
    current, back = start, (start[0]-1,start[1])
    points, first_next = [], None
    for _ in range(8*sum(map(sum,mask))+8):
        bx,by=back[0]-current[0],back[1]-current[1]
        begin = DIRECTIONS.index((bx,by))
        found = None
        for step in range(1,9):
            k=(begin+step)%8; dx,dy=DIRECTIONS[k]
            nx,ny=current[0]+dx,current[1]+dy
            if 0<=ny<len(mask) and 0<=nx<len(mask[0]) and mask[ny][nx]:
                found=(nx,ny)
                pdx,pdy=DIRECTIONS[(k-1)%8]
                new_back=(current[0]+pdx,current[1]+pdy)
                break
        if found is None:
            raise ValueError('Degenerate isolated boundary')
        if first_next is None:
            first_next=found
        elif current==start and found==first_next:
            return points
        points.append(current)
        current,back=found,new_back
    raise ValueError('Boundary did not close')


def resample(points, count=32):
    """Linear interpolation at equal fractions of the closed Euclidean perimeter."""
    radius = max(1, len(points)//128)
    points = [(sum(points[(i+k)%len(points)][0] for k in range(-radius,radius+1))/(2*radius+1),
               sum(points[(i+k)%len(points)][1] for k in range(-radius,radius+1))/(2*radius+1))
              for i in range(len(points))]
    edges=list(zip(points,points[1:]+points[:1]))
    lengths=[hypot(b[0]-a[0],b[1]-a[1]) for a,b in edges]
    perimeter=sum(lengths)
    if perimeter<=0:
        raise ValueError('Zero perimeter')
    result=[]; edge=0; accumulated=0.0
    for i in range(count):
        target=perimeter*i/count
        while edge<len(edges)-1 and accumulated+lengths[edge]<target:
            accumulated+=lengths[edge]; edge+=1
        a,b=edges[edge]; t=(target-accumulated)/lengths[edge]
        result.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
    return result


def chain_code(points):
    """32 octal directions in a rotation-normalized frame, canonical cyclic origin.

    Rotate continuous segment headings BEFORE quantization. This avoids the
    usual failure of differential Freeman codes under non-45-degree rotations.
    Input is an ordered 32-point equal-perimeter polygon.
    """
    if len(points)!=32:
        raise ValueError('Descriptor requires exactly 32 boundary points')
    angles=[atan2(b[1]-a[1],b[0]-a[0]) for a,b in zip(points,points[1:]+points[:1])]
    candidates=[]
    for start in range(32):
        candidate=''.join(str(floor(((angles[(start+i)%32]-angles[start])%(2*pi))/(pi/4)+.5)%8)
                          for i in range(32))
        candidates.append(candidate)
    return min(candidates)


def code_distance(a,b):
    """Reconstruct and compare normalized polygons across cyclic origins.

    Uses only the two 32-digit codes; no hidden image or auxiliary features.
    """
    if len(a)!=32 or len(b)!=32 or any(c not in '01234567' for c in a+b):
        raise ValueError('Expected two 32-digit octal codes')
    def decode(code):
        vertices=[]; x=y=0.0
        for digit in code:
            vertices.append((x,y))
            angle=int(digit)*pi/4
            x+=cos(angle); y+=sin(angle)
        # Quantization can leave a closure residual; distribute it uniformly.
        vertices=[(vx-i*x/32,vy-i*y/32) for i,(vx,vy) in enumerate(vertices)]
        cx=sum(v[0] for v in vertices)/32; cy=sum(v[1] for v in vertices)/32
        centered=[(vx-cx,vy-cy) for vx,vy in vertices]
        scale=sqrt(sum(vx*vx+vy*vy for vx,vy in centered)/32)
        if scale<1e-10:
            raise ValueError('Degenerate decoded chain')
        return [(vx/scale,vy/scale) for vx,vy in centered]
    aa,bb=decode(a),decode(b)
    best=1.0
    for shift in range(32):
        dot=cross=0.0
        for i,(x,y) in enumerate(aa):
            u,v=bb[(i+shift)%32]
            dot+=x*u+y*v; cross+=x*v-y*u
        # Analytic least-squares planar rotation, normalized RMS in [0,1].
        best=min(best,sqrt(max(0.0,2-2*hypot(dot,cross)/32))/2)
    return best



def cluster(codes, threshold=.06):
    """Deterministic complete-link agglomeration prevents similarity chaining."""
    distances={(i,j):code_distance(codes[i],codes[j]) for i in range(len(codes)) for j in range(i+1,len(codes))}
    groups=[[i] for i in range(len(codes))]
    while True:
        candidates=[]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                distance=max(distances[min(a,b),max(a,b)] for a in groups[i] for b in groups[j])
                if distance<=threshold:
                    candidates.append((distance,i,j))
        if not candidates:
            break
        _,i,j=min(candidates)
        groups[i]=sorted(groups[i]+groups[j]); groups.pop(j)
    return sorted(groups,key=lambda g:g[0]),distances
