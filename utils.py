import numpy as np

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


def _reflect(v, n):  # R = V - 2(V·N)N (slide 23 in ray_tracing)
    return v - 2.0 * np.dot(v, n) * n


def build_scene_cache(objects):  # not to rebuild lists every time
    lights = [o for o in objects if isinstance(o, Light)]
    materials = [o for o in objects if isinstance(o, Material)]
    geom = [o for o in objects if hasattr(o, "material_index")]

    return {"lights": lights, "materials": materials, "geom": geom}


def get_material_for_surface(surface, materials):
    idx1 = int(getattr(surface, "material_index", 1))

    if idx1 <= 0 or idx1 > len(materials):
        return Material([1, 1, 1], [0, 0, 0], [0, 0, 0], 10.0, 0.0)

    return materials[idx1 - 1]


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


def intersect_sphere(origin, direction, sphere):  # using Ray-Sphere Intersection II from slide 13
    C = _to_vec3(sphere.position)
    r = float(sphere.radius)

    L = C - origin  # L = O - P_0
    tca = np.dot(L, direction)  # tca = L·V
    d2 = np.dot(L, L) - tca * tca  # |L|^2 - tca^2
    rr = r * r
    if d2 > rr:
        return None, None, None

    thc = np.sqrt(max(rr - d2, 0.0))  # sqrt(r^2 - d^2)
    t0 = tca - thc  # t0 = tca - thc
    t1 = tca + thc  # t1 = tca + thc

    if t0 > EPSILON:
        t = t0
    elif t1 > EPSILON:
        t = t1
    else:
        return None, None, None

    hit = origin + direction * t  # P = P_0 + tV
    n = _normalize(hit - C)  # N = (P - O)/||P - O||

    return t, hit, n


def intersect_plane(origin, direction, plane):
    # Plane: N·P = c
    n = _normalize(_to_vec3(plane.normal))
    denom = np.dot(n, direction)  # N·V
    if abs(denom) < 1e-8:
        return None, None, None

    c = float(plane.offset)
    t = (c - np.dot(n, origin)) / denom  # t = (c - N·P0) / (N·V)
    if t <= EPSILON:
        return None, None, None

    hit = origin + direction * t  # P = P_0 + tV
    return t, hit, n


def intersect_cube(origin, direction, cube):
    c = _to_vec3(cube.position)
    s = float(cube.scale)
    half = 0.5 * s
    bmin = c - half
    bmax = c + half

    tmin = -np.inf
    tmax = np.inf
    hit_axis = -1
    hit_sign = 0.0

    for i in range(3):  # plane for each axis x=0, y=1, z=2
        di = direction[i]
        oi = origin[i]
        if abs(di) < 1e-12:
            if oi < bmin[i] or oi > bmax[i]:
                return None, None, None
            continue

        invd = 1.0 / di
        t1 = (bmin[i] - oi) * invd
        t2 = (bmax[i] - oi) * invd
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

    hit = origin + direction * t  # P = P_0 + tV
    n = np.zeros(3, dtype=np.float64)
    if hit_axis >= 0:
        n[hit_axis] = hit_sign
    n = _normalize(n)
    return t, hit, n


def check_ray_object_intersection(ray_origin, ray_dir, geom, ignore_obj=None):
    closest_t = np.inf
    best = None

    for obj in geom:
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
            best = (p, n, obj, t)

    return best


def _is_occluded(shadow_origin, shadow_dir, max_dist, geom, ignore_obj=None):
    # shadow optimization, stops at first intersection
    for obj in geom:
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


def compute_light_intensity(hit_point, light, geom, scene_settings, rng, ignore_obj=None, shadow_N=1):
    # soft shadow using area light sampling
    hp = _to_vec3(hit_point)
    lp = _to_vec3(light.position)

    to_point = hp - lp
    dist = np.linalg.norm(to_point)
    if dist < 1e-12:
        return 1.0
    w = to_point / dist

    radius = float(light.radius)
    shadow_N = max(1, int(shadow_N))

    if shadow_N == 1 or radius <= 0.0:
        origin = lp + w * EPSILON
        visible = 0.0 if _is_occluded(origin, w, dist, geom, ignore_obj=ignore_obj) else 1.0
        return (1.0 - light.shadow_intensity) + light.shadow_intensity * visible

    tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(w, tmp))
    v = _normalize(np.cross(w, u))

    half = 0.5 * radius
    cell = radius / shadow_N

    # divide into N x N grid,
    # select a random point in each cell
    jitter = rng.random((shadow_N, shadow_N, 2))
    du = (np.arange(shadow_N)[:, None] + jitter[:, :, 0]) * cell - half
    dv = (np.arange(shadow_N)[None, :] + jitter[:, :, 1]) * cell - half

    hits = 0
    total = shadow_N * shadow_N

    for i in range(shadow_N):
        for j in range(shadow_N):
            sample = lp + u * du[i, j] + v * dv[i, j]

            seg = hp - sample
            seg_dist = np.linalg.norm(seg)
            if seg_dist < 1e-12:
                hits += 1  # count rays that hit
                continue

            seg_dir = seg / seg_dist
            origin = sample + seg_dir * EPSILON

            if not _is_occluded(origin, seg_dir, seg_dist, geom, ignore_obj=ignore_obj):
                hits += 1  # count rays that hit

            remaining = total - (i * shadow_N + j + 1)

            if hits == 0 and remaining < total * 0.15:  # early out, most rays are blocked
                pass

    visible_ratio = hits / float(total)  # % of rays that hit the points from the light source
    # light intesity=(1−shadow intensity) + shadow intensity∗(% of rays that hit the points from the light source)
    return (1.0 - light.shadow_intensity) + light.shadow_intensity * visible_ratio


def compute_color_at_intersection(
        ray_origin, ray_dir, scene, scene_settings, depth,
        ignore_obj=None,
):
    if depth > int(scene_settings.max_recursions):
        return _to_vec3(scene_settings.background_color)

    geom = scene["geom"]
    hit = check_ray_object_intersection(ray_origin, ray_dir, geom, ignore_obj=ignore_obj)
    if hit is None:
        return _to_vec3(scene_settings.background_color)

    hit_point, normal_g, obj, _t = hit
    normal_g = _normalize(normal_g)

    front_face = (np.dot(ray_dir, normal_g) < 0.0)
    n_shade = normal_g if front_face else -normal_g

    mat = get_material_for_surface(obj, scene["materials"])  # check material
    view_dir = _normalize(-ray_dir)

    baseN = max(1, int(scene_settings.root_number_shadow_rays))
    shadow_N = baseN if depth == 0 else max(1, baseN // 2)

    rng = scene["_rng"]

    # Local
    local = np.zeros(3, dtype=np.float64)

    for light in scene["lights"]:  # Diffuse + Specular
        L = _to_vec3(light.position) - hit_point
        dist = np.linalg.norm(L)
        if dist < 1e-12:
            continue
        light_dir = L / dist

        # soft shadow
        intensity = compute_light_intensity(
            hit_point, light, geom, scene_settings, rng,
            ignore_obj=obj,
            shadow_N=shadow_N,
        )

        # diffuse
        ndotl = max(np.dot(n_shade, light_dir), 0.0)  # max(N·L, 0)
        diffuse = _to_vec3(mat.diffuse_color) * _to_vec3(light.color) * ndotl * intensity  # Kd * Ip * max(N·L, 0)

        # specular (Phong)
        R = _normalize(_reflect(-light_dir, n_shade))  # R=V-2(V·N)N (slide 23 in ray_tracing)
        spec_angle = max(np.dot(view_dir, R), 0.0)  # R·V
        spec = (spec_angle ** float(mat.shininess)) if ndotl > 0.0 else 0.0  # (R·V)^n
        specular = (_to_vec3(mat.specular_color) * _to_vec3(light.color) *  # Ks * Ip * (R·V)^n
                    spec * float(light.specular_intensity) * intensity)

        local += diffuse + specular  # local color

    C = local

    # reflection / mirror
    kr = _to_vec3(mat.reflection_color)
    reflective = (depth < int(scene_settings.max_recursions)) and np.any(kr > 0.0)

    if reflective:
        refl_dir = _normalize(_reflect(ray_dir, n_shade))
        refl_origin = hit_point + n_shade * EPSILON
        C_refl = compute_color_at_intersection(
            refl_origin, refl_dir, scene, scene_settings, depth + 1, ignore_obj=obj
        )  # recursive call
        C = C + kr * C_refl

    # transparent
    kt = float(mat.transparency)
    transparent = (depth < int(scene_settings.max_recursions)) and (kt > 0.0)

    if transparent:
        offset_n = -normal_g if front_face else normal_g
        behind_origin = hit_point + offset_n * EPSILON
        C_through = compute_color_at_intersection(
            behind_origin, ray_dir, scene, scene_settings, depth + 1, ignore_obj=obj
        )  # recursive call
        C = (1.0 - kt) * C + kt * C_through

    return C


def render_scene(camera, scene_settings, objects, width, height):
    image = np.zeros((height, width, 3), dtype=np.float64)

    scene = build_scene_cache(objects)

    scene["_rng"] = np.random.default_rng(0)  # with seed

    E, P0, step_x, step_y = precompute_camera(camera, width, height)
    p_row_start = P0 + 0.5 * step_x - 0.5 * step_y

    # For each pixel
    for y in range(height):
        p = p_row_start - y * step_y  # move down
        for x in range(width):
            ray_dir = _normalize(p - E)
            color = compute_color_at_intersection(E, ray_dir, scene, scene_settings, depth=0)

            color = np.clip(color, 0.0, 1.0)  # clamp (no clamp within recursion, was changing color)
            image[y, x] = color * 255.0

            p = p + step_x  # move right

    return image
