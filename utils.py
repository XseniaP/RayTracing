# import numpy as np
#
# EPSILON = 1e-4
#
# # ----------------------------
# # Camera helpers
# # ----------------------------
# def discover_pixel_location(position, look_at, up_vector, screen_distance, screen_width, x, y, width, height):
#     position = np.array(position, dtype=float)
#     look_at = np.array(look_at, dtype=float)
#     up_vector = np.array(up_vector, dtype=float)
#
#     forward = look_at - position
#     forward /= np.linalg.norm(forward)
#
#     right = np.cross(forward, up_vector)
#     right /= np.linalg.norm(right)
#
#     up = np.cross(right, forward)
#
#     aspect_ratio = height / width
#     screen_height = screen_width * aspect_ratio
#
#     screen_center = position + forward * screen_distance
#
#     px = (x + 0.5) / width - 0.5
#     py = 0.5 - (y + 0.5) / height
#
#     pixel_position = screen_center + right * (px * screen_width) + up * (py * screen_height)
#     return pixel_position
#
# def construct_ray_from_camera(camera_position, pixel_position):
#     origin = np.array(camera_position, dtype=float)
#     direction = np.array(pixel_position, dtype=float) - origin
#     direction /= np.linalg.norm(direction)
#     return origin, direction
#
# # ----------------------------
# # Object helpers
# # ----------------------------
# def get_geometry_objects(objects):
#     return [o for o in objects if hasattr(o, "material_index")]
#
# def get_lights(objects):
#     from light import Light
#     return [o for o in objects if isinstance(o, Light)]
#
# def get_material(obj, objects):
#     # Surfaces have a material_index starting from 0
#     mat_index = getattr(obj, "material_index", None)
#     if mat_index is None or mat_index >= len(objects):
#         from material import Material
#         return Material([1,1,1], [1,1,1], [0,0,0], 10, 0)
#     mat = objects[mat_index]
#     from material import Material
#     return mat if isinstance(mat, Material) else Material([1,1,1], [1,1,1], [0,0,0], 10, 0)
#
# # ----------------------------
# # Intersections
# # ----------------------------
# def check_ray_object_intersection(ray_origin, ray_dir, objects):
#     closest_t = np.inf
#     closest_hit = None
#
#     for obj in get_geometry_objects(objects):
#         if obj.__class__.__name__ == "Sphere":
#             t, point, normal = intersect_sphere(ray_origin, ray_dir, obj)
#         elif obj.__class__.__name__ == "InfinitePlane":
#             t, point, normal = intersect_plane(ray_origin, ray_dir, obj)
#         elif obj.__class__.__name__ == "Cube":
#             t, point, normal = intersect_cube(ray_origin, ray_dir, obj)
#         else:
#             continue
#
#         if t is not None and EPSILON < t < closest_t:
#             closest_t = t
#             closest_hit = (point, normal, obj)
#
#     return closest_hit
#
# # Sphere intersection
# def intersect_sphere(origin, direction, sphere):
#     L = np.array(sphere.position) - origin
#     r = sphere.radius
#     tca = np.dot(L, direction)
#     d2 = np.dot(L, L) - tca*tca
#     if d2 > r*r:
#         return None, None, None
#     thc = np.sqrt(r*r - d2)
#     t0 = tca - thc
#     t1 = tca + thc
#     t = t0 if t0 > EPSILON else t1
#     if t < EPSILON:
#         return None, None, None
#     hit_point = origin + direction * t
#     normal = (hit_point - np.array(sphere.position)) / r
#     return t, hit_point, normal
#
# # Plane intersection
# def intersect_plane(origin, direction, plane):
#     normal = np.array(plane.normal)
#     denom = np.dot(normal, direction)
#     if abs(denom) < 1e-6:
#         return None, None, None
#     t = (plane.offset - np.dot(normal, origin)) / denom
#     if t < EPSILON:
#         return None, None, None
#     hit_point = origin + direction * t
#     return t, hit_point, normal
#
# # Cube intersection (axis-aligned)
# def intersect_cube(origin, direction, cube):
#     min_corner = np.array(cube.position) - cube.scale/2
#     max_corner = np.array(cube.position) + cube.scale/2
#     tmin = (min_corner - origin) / direction
#     tmax = (max_corner - origin) / direction
#     t1 = np.minimum(tmin, tmax)
#     t2 = np.maximum(tmin, tmax)
#     t_near = np.max(t1)
#     t_far = np.min(t2)
#     if t_near > t_far or t_far < EPSILON:
#         return None, None, None
#     t_hit = t_near if t_near > EPSILON else t_far
#     hit_point = origin + direction * t_hit
#     # Normal
#     epsilon = 1e-5
#     normal = np.zeros(3)
#     for i in range(3):
#         if abs(hit_point[i] - min_corner[i]) < epsilon:
#             normal[i] = -1
#         elif abs(hit_point[i] - max_corner[i]) < epsilon:
#             normal[i] = 1
#     return t_hit, hit_point, normal
#
# # ----------------------------
# # Lighting
# # ----------------------------
# # def compute_color_at_intersection(ray_origin, ray_dir, objects, scene_settings, depth):
# #     hit = check_ray_object_intersection(ray_origin, ray_dir, objects)
# #     if hit is None:
# #         return np.array(scene_settings.background_color)
# #
# #     hit_point, normal, obj = hit
# #     mat = get_material(obj, objects)
# #     view_dir = -ray_dir
# #     color = np.zeros(3)
# #
# #     for light in get_lights(objects):
# #         light_dir = np.array(light.position) - hit_point
# #         dist = np.linalg.norm(light_dir)
# #         light_dir /= dist
# #
# #         # Soft shadow
# #         intensity = compute_light_intensity(hit_point, light, objects)
# #
# #         # Diffuse
# #         diff = max(np.dot(normal, light_dir), 0.0)
# #         diffuse = np.array(mat.diffuse_color) * np.array(light.color) * diff * intensity
# #
# #         # Specular
# #         reflect_dir = 2*np.dot(normal, light_dir)*normal - light_dir
# #         spec = max(np.dot(view_dir, reflect_dir), 0.0) ** mat.shininess
# #         specular = np.array(mat.specular_color) * np.array(light.color) * spec * light.specular_intensity * intensity
# #
# #         color += diffuse + specular
# #
# #     # Reflection
# #     if np.any(mat.reflection_color) and depth < scene_settings.max_recursions:
# #         reflect_dir = ray_dir - 2*np.dot(ray_dir, normal)*normal
# #         reflect_origin = hit_point + EPSILON * reflect_dir
# #         reflected_color = compute_color_at_intersection(reflect_origin, reflect_dir, objects, scene_settings, depth+1)
# #         color += np.array(mat.reflection_color) * reflected_color
# #
# #     return np.clip(color, 0, 1)
#
# def compute_color_at_intersection(ray_origin, ray_dir, objects, scene_settings, depth):
#     hit = check_ray_object_intersection(ray_origin, ray_dir, objects)
#     if hit is None:
#         return np.array(scene_settings.background_color)
#
#     hit_point, normal, obj = hit
#     mat = get_material(obj, objects)
#     view_dir = -ray_dir
#     color = np.zeros(3)
#
#     # ----------------------------
#     # Diffuse + Specular
#     # ----------------------------
#     for light in get_lights(objects):
#         light_dir = np.array(light.position) - hit_point
#         dist = np.linalg.norm(light_dir)
#         light_dir /= dist
#
#         # Soft shadow
#         intensity = compute_light_intensity(hit_point, light, objects)
#
#         # Diffuse
#         diff = max(np.dot(normal, light_dir), 0.0)
#         diffuse = np.array(mat.diffuse_color) * np.array(light.color) * diff * intensity
#
#         # Specular
#         reflect_dir = 2*np.dot(normal, light_dir)*normal - light_dir
#         spec = max(np.dot(view_dir, reflect_dir), 0.0) ** mat.shininess
#         specular = np.array(mat.specular_color) * np.array(light.color) * spec * light.specular_intensity * intensity
#
#         color += diffuse + specular
#
#     # ----------------------------
#     # Reflection
#     # ----------------------------
#     if np.any(mat.reflection_color) and depth < scene_settings.max_recursions:
#         reflect_dir = ray_dir - 2*np.dot(ray_dir, normal)*normal
#         reflect_origin = hit_point + EPSILON * reflect_dir
#         reflected_color = compute_color_at_intersection(reflect_origin, reflect_dir, objects, scene_settings, depth+1)
#         color += np.array(mat.reflection_color) * reflected_color
#
#     # ----------------------------
#     # Transparency (shoot ray through surface)
#     # ----------------------------
#     if mat.transparency > 0 and depth < scene_settings.max_recursions:
#         trans_origin = hit_point + EPSILON * ray_dir
#         trans_color = compute_color_at_intersection(trans_origin, ray_dir, objects, scene_settings, depth+1)
#         # Mix with current color
#         color = mat.transparency * trans_color + (1 - mat.transparency) * color
#
#     return np.clip(color, 0, 1)
#
#
# def compute_light_intensity(hit_point, light, objects):
#     from random import random
#     N = int(np.sqrt(light.radius)) if light.radius > 0 else 1
#     if N <= 1:
#         visible = check_light_visibility(hit_point, light, objects)
#         return (1 - light.shadow_intensity) + light.shadow_intensity * visible
#
#     hits = 0
#     total = N*N
#     light_dir = hit_point - np.array(light.position)
#     light_dir /= np.linalg.norm(light_dir)
#     # Plane for soft shadow
#     temp = np.array([1,0,0]) if abs(light_dir[0]) < 0.9 else np.array([0,1,0])
#     u = np.cross(light_dir, temp)
#     u /= np.linalg.norm(u)
#     v = np.cross(light_dir, u)
#     cell = light.radius / N
#     half = light.radius / 2
#     for i in range(N):
#         for j in range(N):
#             du = (i + random())*cell - half
#             dv = (j + random())*cell - half
#             sample = np.array(light.position) + u*du + v*dv
#             shadow_dir = hit_point - sample
#             shadow_dist = np.linalg.norm(shadow_dir)
#             shadow_dir /= shadow_dist
#             origin = sample + EPSILON*shadow_dir
#             blocked = False
#             for obj in get_geometry_objects(objects):
#                 t_result = check_ray_object_intersection(origin, shadow_dir, [obj])
#                 if t_result is not None:
#                     t_hit = np.linalg.norm(t_result[0]-origin)
#                     if t_hit < shadow_dist - EPSILON:
#                         blocked = True
#                         break
#             if not blocked:
#                 hits += 1
#     return (1 - light.shadow_intensity) + light.shadow_intensity * (hits / total)
#
# def check_light_visibility(point, light, objects):
#     dir = np.array(light.position) - point
#     dist = np.linalg.norm(dir)
#     dir /= dist
#     origin = point + EPSILON*dir
#     for obj in get_geometry_objects(objects):
#         t_result = check_ray_object_intersection(origin, dir, [obj])
#         if t_result is not None:
#             t_hit = np.linalg.norm(t_result[0]-origin)
#             if t_hit < dist - EPSILON:
#                 return 0.0
#     return 1.0
#
# # ----------------------------
# # Render
# # ----------------------------
# def render_scene(camera, scene_settings, objects, width, height):
#     image = np.zeros((height, width, 3))
#     for y in range(height):
#         for x in range(width):
#             pixel = discover_pixel_location(camera.position, camera.look_at, camera.up_vector,
#                                            camera.screen_distance, camera.screen_width, x, y, width, height)
#             origin, ray_dir = construct_ray_from_camera(camera.position, pixel)
#             color = compute_color_at_intersection(origin, ray_dir, objects, scene_settings, 0)
#             image[y, x] = np.clip(color*255, 0, 255)
#     return image


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


def discover_pixel_location(position, look_at, up_vector, screen_distance, screen_width, x, y, width, height):
    # Vz = camera forward direction , Vx = right direction, Vy = up direction
    pos = _to_vec3(position) # Eye/camera position E
    look = _to_vec3(look_at)
    up0 = _to_vec3(up_vector)

    forward = _normalize(look - pos) #Vz

    right = np.cross(up0, forward) #Vx
    right = _normalize(right)

    up = np.cross(forward, right) #Vy
    up = _normalize(up)

    aspect = float(height) / float(width)
    screen_height = screen_width * aspect

    # P = E + Vz * f
    screen_center = pos + forward * float(screen_distance)

    # Pixel coordinates mapped to [-0.5, 0.5]
    px = (x + 0.5) / float(width) - 0.5
    py = 0.5 - (y + 0.5) / float(height)

    # P0 = P - w*Vx - h*Vy or P = P0 + px*w*Vx + py*h*Vy
    return screen_center + right * (px * screen_width) + up * (py * screen_height)

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

def check_ray_object_intersection(ray_origin, ray_dir, objects):
    # Ray-Scene Intersection silde 22 in ray_casting
    closest_t = np.inf
    min_primitive = None

    for obj in get_geometry_objects(objects):
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

    if np.dot(n, direction) > 0:
        n = -n
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
            hit_sign = -1.0 if t1 > t2 else 1.0

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

def compute_color_at_intersection(ray_origin, ray_dir, objects, scene_settings, depth):
    #behind*transparency + local*(1-transparency) + reflection

    if depth > int(scene_settings.max_recursions):
        return np.clip(_to_vec3(scene_settings.background_color), 0.0, 1.0)

    hit = check_ray_object_intersection(ray_origin, ray_dir, objects) #find the nearest intersection of the ray
    if hit is None:
        return np.clip(_to_vec3(scene_settings.background_color), 0.0, 1.0)

    hit_point, normal, obj, _t = hit
    normal = _normalize(normal)

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


        intensity = compute_light_intensity(hit_point, light, objects, scene_settings, ignore_obj=obj) # soft shadow

        # Diffuse I_diff = Kd Ip (N*L)
        ndotl = max(np.dot(normal, light_dir), 0.0)
        diffuse = _to_vec3(mat.diffuse_color) * _to_vec3(light.color) * ndotl * intensity

        # Specular (Phong), I_spec=Ks Ip (R·V)^n
        R = _normalize(_reflect(-light_dir, normal)) # R = (2(L·N)N - L)
        spec_angle = max(np.dot(view_dir, R), 0.0) # (R·V)
        spec = (spec_angle ** float(mat.shininess)) if ndotl > 0.0 else 0.0  #(R·V)^n
        specular = _to_vec3(mat.specular_color) * _to_vec3(light.color) * spec * float(light.specular_intensity) * intensity

        local_color += diffuse + specular # local color

    if depth < int(scene_settings.max_recursions) and np.any(np.array(mat.reflection_color, dtype=np.float64) > 0.0): # if mirror / reflective
        refl_dir = _normalize(_reflect(ray_dir, normal)) #direction symmetric wrt normal
        refl_origin = hit_point + refl_dir * EPSILON
        refl_hit_color = compute_color_at_intersection(refl_origin, refl_dir, objects, scene_settings, depth + 1) # recursive call
        reflection_color = _to_vec3(mat.reflection_color) * refl_hit_color

    trans = float(mat.transparency)
    if trans > 0.0 and depth < int(scene_settings.max_recursions): #if transparent
        behind_origin = hit_point + ray_dir * EPSILON
        behind_color = compute_color_at_intersection(behind_origin, ray_dir, objects, scene_settings, depth + 1) # recursive call

        # output color=(background color)·transparency+(diffuse + specular)·(1−transparency)+(reflection color)
        out = behind_color * trans + local_color * (1.0 - trans) + reflection_color
    else: # output color=(diffuse + specular)+(reflection color)
        out = local_color + reflection_color

    return np.clip(out, 0.0, 1.0)

def render_scene(camera, scene_settings, objects, width, height):
    image = np.zeros((height, width, 3), dtype=np.float64)
    cam_pos = _to_vec3(camera.position)

    # For each pixel
    for y in range(height):
        for x in range(width):
            pixel_pos = discover_pixel_location(
                camera.position, camera.look_at, camera.up_vector,
                camera.screen_distance, camera.screen_width,
                x, y, width, height
            )
            origin, ray_dir = construct_ray_from_camera(cam_pos, pixel_pos) #incl finding first intersection
            color_out = compute_color_at_intersection(origin, ray_dir, objects, scene_settings, depth=0)
            image[y, x] = np.clip(color_out * 255.0, 0.0, 255.0)

    return image
