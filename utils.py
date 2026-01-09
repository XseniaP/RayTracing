import numpy as np
import random

from material import Material
from light import Light

EPSILON = 0.0001

def _to_vec3(x):
    return np.array(x, dtype=np.float64)

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _reflect(v, n): # R = V - 2(V·N)N (slide 23 in ray_tracing)
    return v - 2.0 * np.dot(v, n) * n


def precompute_camera(camera, width, height):
    E = _to_vec3(camera.position)
    A = _to_vec3(camera.look_at)
    up0 = _to_vec3(camera.up_vector)

    Vz = _normalize(A - E)
    Vx = _normalize(np.cross(up0, Vz))
    Vy = _normalize(np.cross(Vz, Vx))

    f = float(camera.screen_distance)

    screen_w = float(camera.screen_width)
    aspect = float(height) / float(width)
    screen_h = screen_w * aspect

    w = 0.5 * screen_w
    h = 0.5 * screen_h

    P = E + Vz * f
    P0 = P - w * Vx + h * Vy  # top-left

    step_x = (screen_w / float(width)) * Vx
    step_y = (screen_h / float(height)) * Vy

    return E, P0, step_x, step_y

def construct_ray_from_camera(camera_position, pixel_position): # P = P0+tV
    origin = _to_vec3(camera_position)
    direction = _normalize(_to_vec3(pixel_position) - origin)
    return origin, direction

def get_lights(objects):
    return [o for o in objects if isinstance(o, Light)]

def get_materials(objects):
    return [o for o in objects if isinstance(o, Material)]

def get_geometry_objects(objects):
    return [o for o in objects if hasattr(o, "material_index")]

def get_material_for_surface(surface, objects):
    mats = get_materials(objects)
    idx1 = int(getattr(surface, "material_index", 0))
    if idx1 <= 0 or idx1 > len(mats):
        return Material([1, 1, 1], [0, 0, 0], [0, 0, 0], 10.0, 0.0)
    return mats[idx1 - 1]

def check_ray_object_intersection(ray_origin, ray_dir, objects, ignore_obj=None):
    # Ray-Scene Intersection slide 22 in ray_casting
    closest_t = np.inf
    min_primitive = None

    for obj in get_geometry_objects(objects):
        if ignore_obj is not None and obj is ignore_obj:
            continue
        name = obj.__class__.__name__
        if name == "Sphere":
            t, p, n = intersect_sphere(ray_origin, ray_dir, obj)
        elif name == "InfinitePlane":
            t, p, n = intersect_plane(ray_origin, ray_dir, obj)
        elif name == "Cube":
            t, p, n = intersect_cube(ray_origin, ray_dir, obj)
        else:
            continue

        if t is not None and EPSILON < t < closest_t:
            closest_t = t
            min_primitive = (p, n, obj, t)

    return min_primitive

def intersect_sphere(origin, direction, sphere): # using Ray-Sphere Intersection II from slide 13
    C = _to_vec3(sphere.position)
    r = float(sphere.radius)

    L = C - origin  # L = O - P_0
    tca = np.dot(L, direction) # tca = L·V
    d_square = np.dot(L, L) - tca * tca #|L|^2 - tca^2
    rr = r * r
    if d_square > rr:
        return None, None, None

    thc = np.sqrt(max(rr - d_square, 0.0)) # sqrt(r^2 - d^2)
    t0 = tca - thc # t0 = tca - thc
    t1 = tca + thc # t1 = tca + thc

    t = None
    if t0 > EPSILON:
        t = t0
    elif t1 > EPSILON:
        t = t1
    else:
        return None, None, None

    hit = origin + direction * t #P = P_0 + tV
    n = _normalize(hit - C) #N = (P - O)/||P - O||
    return t, hit, n

def intersect_plane(origin, direction, plane):
    # N·P = c
    # P = P0 + t V
    # t = (c - N·P0) / (N·V)
    n = _normalize(_to_vec3(plane.normal))
    denom = np.dot(n, direction) #N·V
    if abs(denom) < 1e-8:
        return None, None, None

    c = float(plane.offset)
    t = (c - np.dot(n, origin)) / denom # t = (c - N·P0) / (N·V)
    if t <= EPSILON:
        return None, None, None

    hit = origin + direction * t #P = P_0 + tV

    return t, hit, n

def intersect_cube(origin, direction, cube): #Intersect 3 front-facing planes, return closest
    c = _to_vec3(cube.position)
    s = float(cube.scale)
    half = 0.5 * s
    bmin = c - half
    bmax = c + half

    tmin = -np.inf
    tmax = np.inf
    hit_axis = -1
    hit_sign = 0.0

    for i in range(3): # plane for each axis x=0, y=1, z=2
        if abs(direction[i]) < 1e-12:
            if origin[i] < bmin[i] or origin[i] > bmax[i]:
                return None, None, None
            continue

        invd = 1.0 / direction[i]
        t1 = (bmin[i] - origin[i]) * invd
        t2 = (bmax[i] - origin[i]) * invd
        near = min(t1, t2)
        far = max(t1, t2)

        if near > tmin:
            tmin = near
            hit_axis = i
            hit_sign = -1.0 if t1 < t2 else 1.0

        tmax = min(tmax, far)
        if tmin > tmax:
            return None, None, None

    t = tmin if tmin > EPSILON else tmax
    if t <= EPSILON:
        return None, None, None

    hit = origin + direction * t #P = P_0 + tV
    n = np.zeros(3, dtype=np.float64)
    if hit_axis >= 0:
        n[hit_axis] = hit_sign
    n = _normalize(n)
    return t, hit, n

def _is_occluded(shadow_origin, shadow_dir, max_dist, objects, ignore_obj=None):
    # shadow optimization, stops at first intersection
    for obj in get_geometry_objects(objects):
        if ignore_obj is not None and obj is ignore_obj:
            continue

        name = obj.__class__.__name__
        if name == "Sphere":
            t, _, _ = intersect_sphere(shadow_origin, shadow_dir, obj)
        elif name == "InfinitePlane":
            t, _, _ = intersect_plane(shadow_origin, shadow_dir, obj)
        elif name == "Cube":
            t, _, _ = intersect_cube(shadow_origin, shadow_dir, obj)
        else:
            continue

        if t is not None and EPSILON < t < (max_dist - EPSILON):
            return True

    return False

def compute_light_intensity(hit_point, light, objects, scene_settings, ignore_obj=None):
    # soft shadow using area light sampling

    N = int(scene_settings.root_number_shadow_rays)
    N = max(1, N)

    hp = _to_vec3(hit_point)
    lp = _to_vec3(light.position)

    to_point = hp - lp
    dist = np.linalg.norm(to_point)
    if dist < 1e-12:
        return 1.0
    w = to_point / dist

    tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(w, tmp))
    v = _normalize(np.cross(w, u))

    radius = float(light.radius)
    if N == 1 or radius <= 0.0:
        shadow_origin = lp + w * EPSILON
        occluded = _is_occluded(shadow_origin, w, dist, objects, ignore_obj=ignore_obj)
        visible = 0.0 if occluded else 1.0
        return (1.0 - light.shadow_intensity) + light.shadow_intensity * visible

    half_radius = radius * 0.5
    cell = radius / N

    hits = 0
    total = N * N

    # divide into N x N grid,
    # select a random point in each cell (uniformly sample x value and y value) to avoid banding artifacts
    for i in range(N):
        for j in range(N):
            du = (i + random.random()) * cell - half_radius
            dv = (j + random.random()) * cell - half_radius
            sample = lp + u * du + v * dv

            seg = hp - sample
            seg_dist = np.linalg.norm(seg)
            if seg_dist < 1e-12:
                hits += 1 #count rays that hit
                continue

            seg_dir = seg / seg_dist
            origin = sample + seg_dir * EPSILON

            if not _is_occluded(origin, seg_dir, seg_dist, objects, ignore_obj=ignore_obj):
                hits += 1

    visible_ratio = hits / float(total) # % of rays that hit the points from the light source
    # light intesity=(1−shadow intensity)∗1 +shadow intensity∗(% of rays that hit the points from the light source)
    return (1.0 - light.shadow_intensity) + light.shadow_intensity * visible_ratio

def compute_color_at_intersection(ray_origin, ray_dir, objects, scene_settings, depth, ignore_obj=None, allow_shadows=True):

    if depth > int(scene_settings.max_recursions):
        return _to_vec3(scene_settings.background_color)

    hit = check_ray_object_intersection(ray_origin, ray_dir, objects, ignore_obj=ignore_obj) #find the nearest intersection of the ray
    if hit is None:
        return _to_vec3(scene_settings.background_color)

    hit_point, normal, obj, _t = hit
    normal = _normalize(normal)
    front_face = np.dot(ray_dir, normal) < 0.0
    shading_normal = normal if front_face else -normal

    mat = get_material_for_surface(obj, objects) #check material
    view_dir = _normalize(-ray_dir)

    local_color = np.zeros(3, dtype=np.float64)
    reflection_color = np.zeros(3, dtype=np.float64)

    for light in get_lights(objects): # Diffuse + Specular
        L = _to_vec3(light.position) - hit_point
        dist = np.linalg.norm(L)
        if dist < 1e-12:
            continue
        light_dir = L / dist

        intensity = compute_light_intensity(hit_point, light, objects, scene_settings, ignore_obj=obj) if allow_shadows else 1.0 # soft shadow

        # Diffuse I_diff = Kd Ip (N*L)
        ndotl = max(np.dot(shading_normal, light_dir), 0.0)
        diffuse = _to_vec3(mat.diffuse_color) * _to_vec3(light.color) * ndotl * intensity

        # Specular (Phong), I_spec=Ks Ip (R·V)^n
        R = _normalize(_reflect(-light_dir, shading_normal)) # R = (2(L·N)N - L)
        spec_angle = max(np.dot(view_dir, R), 0.0) # (R·V)
        spec = (spec_angle ** float(mat.shininess)) if ndotl > 0.0 else 0.0  #(R·V)^n
        specular = _to_vec3(mat.specular_color) * _to_vec3(light.color) * spec * float(light.specular_intensity) * intensity

        local_color += diffuse + specular # local color

    if depth < int(scene_settings.max_recursions) and np.any(np.array(mat.reflection_color, dtype=np.float64) > 0.0): # if mirror / reflective
        refl_dir = _normalize(_reflect(ray_dir, shading_normal)) #direction symmetric wrt normal
        refl_origin = hit_point + shading_normal * EPSILON
        refl_hit_color = compute_color_at_intersection(refl_origin, refl_dir, objects, scene_settings, depth + 1, ignore_obj=obj, allow_shadows=False) # recursive call
        reflection_color = _to_vec3(mat.reflection_color) * refl_hit_color

    trans = float(mat.transparency)
    if trans > 0.0 and depth < int(scene_settings.max_recursions): #if transparent
        offset_normal = -normal if front_face else normal

        behind_origin = hit_point + offset_normal * EPSILON
        behind_color = compute_color_at_intersection(behind_origin, ray_dir, objects, scene_settings, depth + 1, ignore_obj=obj) # recursive call

        # output color=(background color)·transparency + (diffuse + specular)·(1−transparency) + (reflection color)
        out = behind_color * trans + local_color * (1.0 - trans) + reflection_color
    else: # output color=(diffuse + specular)+(reflection color)
        out = local_color + reflection_color

    return np.maximum(out, 0.0)

def render_scene(camera, scene_settings, objects, width, height):
    image = np.zeros((height, width, 3), dtype=np.float64)

    E, P0, step_x, step_y = precompute_camera(camera, width, height)
    p_row_start = P0 + 0.5 * step_x - 0.5 * step_y

    # For each pixel
    for y in range(height):
        p = p_row_start - y * step_y # move down
        for x in range(width):
            ray_dir = _normalize(p - E)
            color_out = compute_color_at_intersection(E, ray_dir, objects, scene_settings, depth=0)
            color_out = np.clip(color_out, 0.0, 1.0)
            image[y, x] = np.clip(color_out * 255.0, 0.0, 255.0)
            p = p + step_x  # move right

    return image
