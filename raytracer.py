import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import NamedTuple

# Plot Settings
plot_xmin = -3.0
plot_xmax = 3.0
plot_ymin = -3.0
plot_ymax = 3.0
plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.set_xlim([plot_xmin, plot_xmax])
ax.set_ylim([plot_ymin, plot_ymax])
ax.set_aspect('equal')
scale = 1.0

# Preallocate line storage
max_depth = 5  # raytrace depth
max_total = 10000
max_rays = 1000
max_angles = 360
line_dtype = np.dtype([
    ('start', np.float64, 2),
    ('end', np.float64, 2),
    ('color', np.float64, 3),
    ('alpha', np.float64)
])
all_lines = np.zeros(2 ** (max_depth + 1) * max_total, dtype=line_dtype)
line_index = 0

# Image compositing settings
img_width = 1024
img_height = 1024

# Mouse and keyboard event definitions
click = False
shift = False
right_click = False
light_selected = False
light_start_selected = False
light_end_selected = False
sphere_selected = False
sphere_edge_selected = False
click_position = np.array([])
edge_selection_tolerance = 0.2
light_start = np.array([])
light_center = np.array([])
light_end = np.array([])

# Store rays to be raytraced
class RayStruct(NamedTuple):
    """Ray structure"""
    origin: np.ndarray
    direction: np.ndarray
    pol: np.ndarray
    wavelength: np.float32


def onclick(event):
    """Mouse click event"""
    global click
    if event.inaxes == ax:
        if event.button == 1:
            click = True
            xpos_slider.set_val(event.xdata)
            ypos_slider.set_val(event.ydata)
        if event.button == 3:
            pass


def onrelease(event):
    """Mouse release event"""
    global click
    if event.inaxes == ax:
        click = False
        right_click = False


def onmove(event):
    """Mouse move event"""
    global click, shift, right_click
    if event.inaxes == ax and click and not shift:
        xpos_slider.set_val(event.xdata)
        ypos_slider.set_val(event.ydata)
    if event.inaxes == ax and click and shift:
        start = np.array([xpos_slider.val, ypos_slider.val])
        end = np.array([event.xdata, event.ydata])
        direction = end - start
        length = np.linalg.norm(direction)
        if length:
            direction = direction / length
            angle = np.degrees(np.arctan2(direction[0], direction[1]))
            if angle:
                rot_slider.set_val(angle)


def on_key_press(event):
    """Mouse key press event"""
    global shift, render
    if event.key == 'shift':
        shift = True
    if event.key == 'p':
        render = True
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig("image.png", bbox_inches=extent)
    print(shift)


def on_key_release(event):
    """Mouse key release event"""
    global shift
    if event.key == 'shift':
        shift = False


def on_mouse_wheel(event):
    """Mouse wheel event"""
    global scale, plot_xmin, plot_xmax, plot_ymin, plot_ymax, ax
    if event.inaxes:
        scale_step = 1.1
        if event.button == 'up':
            scale /= scale_step
        if event.button == 'down':
            scale *= scale_step
        if scale > 10:
            scale = 10
        if scale < 0.1:
            scale = 0.1
        zoom_slider.set_val(scale)


@jit(nopython=True, cache=True)
def get_color(wavelength):
    """Convert wavelength in nm to an RGB array."""
    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    # Intensity correction near vision limits
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength <= 700:
        factor = 1.0
    elif 700 < wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0

    return np.array([R * factor, G * factor, B * factor])


@jit(nopython=True, cache=True)
def get_ior(wavelength, cauchy_a, cauchy_b):
    """Get IOR for wavelength [nm] using cauchy coefficients A and B[um^2]"""
    lambda2 = wavelength * wavelength
    nm2_per_um2 = 1000 * 1000
    return cauchy_a + cauchy_b * nm2_per_um2 / lambda2


@jit(nopython=True, cache=True)
def rotate(vector, theta_deg):
    """Rotate vector by theta degrees"""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


@jit(nopython=True, cache=True)
def reflection(vector, normal):
    """Calculate reflected ray from incident ray and normal"""
    return vector - 2 * np.dot(vector, normal) * normal


@jit(nopython=True, cache=True)
def refraction(vector, normal, n1, n2):
    """Calculate refracted ray from incident ray, normal, and indices of refraction"""
    cos_theta_i = -np.dot(vector, normal)
    sin_theta_i = np.sqrt(1.0 - cos_theta_i ** 2)

    sin_theta_t = (n1 / n2) * sin_theta_i

    if sin_theta_t > 1.0:
        return np.array([np.nan, np.nan])

    cos_theta_t = np.sqrt(1.0 - sin_theta_t ** 2)
    refr = (n1 / n2) * vector + ((n1 / n2) * cos_theta_i - cos_theta_t) * normal

    return refr


@jit(nopython=True, cache=True)
def polarization(vector, normal, ni, nt):
    """Get Fresnel reflection and transmission coefficients (Rs, Rp, Ts, Tp)"""
    cos_theta_i = -np.dot(vector, normal)
    sin_theta_i = np.sqrt(max(0.0, 1.0 - cos_theta_i**2))

    sin_theta_t = (ni / nt) * sin_theta_i

    if sin_theta_t > 1.0:  # Total internal reflection
        return np.array([1.0, 1.0]), np.array([0.0, 0.0])

    cos_theta_t = np.sqrt(max(0.0, 1.0 - sin_theta_t**2))

    Rs = ((ni * cos_theta_i - nt * cos_theta_t) /
          (ni * cos_theta_i + nt * cos_theta_t))**2
    Rp = ((ni * cos_theta_t - nt * cos_theta_i) /
          (ni * cos_theta_t + nt * cos_theta_i))**2

    Ts = (4 * ni * nt * cos_theta_i * cos_theta_t) / \
         (ni * cos_theta_i + nt * cos_theta_t)**2
    Tp = (4 * ni * nt * cos_theta_i * cos_theta_t) / \
         (ni * cos_theta_t + nt * cos_theta_i)**2

    return np.array([Rs, Rp]), np.array([Ts, Tp])


@jit(nopython=True, cache=True)
def ray_circle_intersection(origin, direction, sphere_center, sphere_radius):
    """Get intersection points of ray with sphere"""
    oc = sphere_center - origin
    a = np.dot(direction, direction)
    h = np.dot(direction, oc)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius

    discriminant = h * h - a * c

    if discriminant < 0:
        return np.array([np.nan, np.nan])
    else:
        distance1 = (h - np.sqrt(discriminant)) / a
        if distance1 > 0.0001:
            return origin + distance1 * direction

        distance2 = (h + np.sqrt(discriminant)) / a
        if distance2 > 0.0001:
            return origin + distance2 * direction

        return np.array([np.nan, np.nan])


@jit(nopython=True, cache=True)
def trace_single_ray(origin, direction, pol, wavelength, n1, n2, sphere_center, sphere_radius, max_depth, lines_out,
                     line_idx):
    """Iterative raytracing for a single initial ray using a stack-based approach."""
    ray_extension = 100.0

    stack_size = 2 ** (max_depth + 1)
    stack_origins = np.empty((stack_size, 2), dtype=np.float64)
    stack_directions = np.empty((stack_size, 2), dtype=np.float64)
    stack_pols = np.empty((stack_size, 2), dtype=np.float64)
    stack_wavelengths = np.empty(stack_size, dtype=np.float64)
    stack_depths = np.empty(stack_size, dtype=np.int32)
    stack_is_inside = np.empty(stack_size, dtype=np.bool_)
    stack_ptr = 0

    stack_origins[0] = origin
    stack_directions[0] = direction
    stack_pols[0] = pol
    stack_wavelengths[0] = wavelength
    stack_depths[0] = 0
    stack_is_inside[0] = False
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        curr_origin = stack_origins[stack_ptr]
        curr_direction = stack_directions[stack_ptr]
        curr_pol = stack_pols[stack_ptr]
        curr_wavelength = stack_wavelengths[stack_ptr]
        curr_depth = stack_depths[stack_ptr]
        is_inside = stack_is_inside[stack_ptr]

        if curr_depth > max_depth:
            continue

        if np.isnan(curr_direction[0]) or np.isnan(curr_direction[1]):
            continue

        alpha = np.mean(curr_pol)
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        color = get_color(curr_wavelength)

        ray_end = ray_circle_intersection(curr_origin, curr_direction, sphere_center, sphere_radius)

        no_intersect = np.isnan(ray_end[0])
        dist_from_center = np.sqrt((curr_origin[0] - sphere_center[0]) ** 2 + (curr_origin[1] - sphere_center[1]) ** 2)
        curr_is_inside = dist_from_center - 0.001 < sphere_radius

        test_point = curr_origin + curr_direction * 0.001
        dist_test = np.sqrt((test_point[0] - sphere_center[0]) ** 2 + (test_point[1] - sphere_center[1]) ** 2)
        is_directed_outward = dist_test > sphere_radius

        if no_intersect or (curr_is_inside and is_directed_outward):
            if line_idx[0] < len(lines_out):
                lines_out[line_idx[0]]['start'] = curr_origin
                lines_out[line_idx[0]]['end'] = curr_origin + curr_direction * ray_extension
                lines_out[line_idx[0]]['color'] = color
                lines_out[line_idx[0]]['alpha'] = alpha
                line_idx[0] += 1
            continue

        if line_idx[0] < len(lines_out):
            lines_out[line_idx[0]]['start'] = curr_origin
            lines_out[line_idx[0]]['end'] = ray_end
            lines_out[line_idx[0]]['color'] = color
            lines_out[line_idx[0]]['alpha'] = alpha
            line_idx[0] += 1

        normal = ray_end - sphere_center
        normal_len = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
        normal = normal / normal_len

        if curr_is_inside:
            curr_n1 = n2
            curr_n2 = n1
            normal = -normal
        else:
            curr_n1 = n1
            curr_n2 = n2

        R, T = polarization(curr_direction, normal, curr_n1, curr_n2)

        v_refr = refraction(curr_direction, normal, curr_n1, curr_n2)
        v_refl = reflection(curr_direction, normal)

        if not np.isnan(v_refr[0]) and stack_ptr < stack_size:
            new_pol = curr_pol * T
            stack_origins[stack_ptr] = ray_end
            stack_directions[stack_ptr] = v_refr
            stack_pols[stack_ptr] = new_pol
            stack_wavelengths[stack_ptr] = curr_wavelength
            stack_depths[stack_ptr] = curr_depth + 1
            stack_is_inside[stack_ptr] = not curr_is_inside
            stack_ptr += 1

        if not np.isnan(v_refl[0]) and stack_ptr < stack_size:
            new_pol = curr_pol * R
            stack_origins[stack_ptr] = ray_end
            stack_directions[stack_ptr] = v_refl
            stack_pols[stack_ptr] = new_pol
            stack_wavelengths[stack_ptr] = curr_wavelength
            stack_depths[stack_ptr] = curr_depth + 1
            stack_is_inside[stack_ptr] = curr_is_inside
            stack_ptr += 1

    return line_idx[0]


@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def draw_line_additive(img, x0, y0, x1, y1, color, alpha, line_width):
    """Draw line with additive blending (work around for MacOS)"""
    height, width = img.shape[0], img.shape[1]

    # Quick bounds check - skip if completely outside
    if (max(x0, x1) < 0 or min(x0, x1) >= width or
            max(y0, y1) < 0 or min(y0, y1) >= height):
        return

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    if dx == 0 and dy == 0:
        return

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    x, y = x0, y0

    # Optimize line width
    line_rad = max(line_width / 2.0, 0.5)
    line_rad_int = int(line_rad) + 1
    line_rad_sq = line_rad * line_rad  # Pre-compute for distance checks

    # Pre-compute color * alpha
    r_contrib = color[0] * alpha
    g_contrib = color[1] * alpha
    b_contrib = color[2] * alpha

    while True:
        # Draw pixels around the center line for thickness
        for offset_x in range(-line_rad_int, line_rad_int + 1):
            px = x + offset_x
            if px < 0 or px >= width:
                continue

            for offset_y in range(-line_rad_int, line_rad_int + 1):
                py = y + offset_y
                if py < 0 or py >= height:
                    continue

                # Calculate distance from line center for anti-aliasing
                dist_sq = float(offset_x * offset_x + offset_y * offset_y)
                if dist_sq > line_rad_sq:
                    continue

                # Smooth falloff at edge
                dist = np.sqrt(dist_sq)
                intensity = max(0.0, 1.0 - (dist / line_rad))

                # Additive blending
                img[py, px, 0] += r_contrib
                img[py, px, 1] += g_contrib
                img[py, px, 2] += b_contrib

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


@jit(nopython=True, cache=True)
def render_lines_to_image(lines_data, line_count, img,
                          plot_xmin, plot_xmax, plot_ymin, plot_ymax,
                          scale, line_width):
    """Render all lines to image buffer with additive blending"""
    height, width = img.shape[0], img.shape[1]

    # Calculate plot bounds
    xmin = plot_xmin * scale
    xmax = plot_xmax * scale
    ymin = plot_ymin * scale
    ymax = plot_ymax * scale

    for i in range(line_count):
        start = lines_data[i]['start']
        end = lines_data[i]['end']
        color = lines_data[i]['color']
        alpha = lines_data[i]['alpha']

        # Convert world coordinates to pixel coordinates (no offset)
        x0 = int((start[0] - xmin) / (xmax - xmin) * width)
        y0 = int((start[1] - ymin) / (ymax - ymin) * height)
        x1 = int((end[0] - xmin) / (xmax - xmin) * width)
        y1 = int((end[1] - ymin) / (ymax - ymin) * height)

        # Flip Y coordinate (matplotlib has origin at bottom-left)
        y0 = height - 1 - y0
        y1 = height - 1 - y1

        # Draw the line
        draw_line_additive(img, x0, y0, x1, y1, color, alpha, line_width)


def update(val):
    """Main update function updates drawing whenever sliders change"""
    global all_lines, line_index, line_width
    line_index = 0

    intensity_val = brightness_slider.val
    xpos_val = xpos_slider.val
    ypos_val = ypos_slider.val
    rot_val = -rot_slider.val
    n_rays_val = int(nrays_slider.val)
    beam_width_val = beam_width_slider.val
    beam_angle_val = beam_angle_val_slider.val
    n_angles_val = int(angles_slider.val)
    n_colors_val = int(colors_slider.val)
    sphere_cauchy_a = sphere_cauchy_a_slider.val
    sphere_cauchy_b = sphere_cauchy_b_slider.val
    medium_cauchy_a = medium_cauchy_a_slider.val
    medium_cauchy_b = medium_cauchy_b_slider.val
    polarization_val = polarization_slider.val
    sphere_radius_val = sphere_radius_slider.val
    line_width_val = line_width_slider.val
    wavelength_val = wavelength_slider.val
    depth_val = depth_slider.val
    sphere_xpos = sphere_xpos_slider.val
    sphere_ypos = sphere_ypos_slider.val

    nrays = n_rays_val * n_colors_val * n_angles_val
    sphere_center = np.array([sphere_xpos, sphere_ypos])

    if nrays > max_total:
        ax.set_xlabel("WARNING: Ray count is too high! Try reducing number of rays. " + str(nrays))
        return fig,

    line_width = line_width_val

    if beam_width_val == 0:
        n_rays_val = 1

    # Ray starting positions
    if n_rays_val <= 1:
        beam_width_val = 0
        x_arr = np.array([xpos_val])
        y_arr = np.array([ypos_val])
    else:
        if rot_val != 0:
            r_start = rotate(np.array([-beam_width_val / 2, 0]), rot_val) + np.array([xpos_val, ypos_val])
            r_end = rotate(np.array([beam_width_val / 2, 0]), rot_val) + np.array([xpos_val, ypos_val])
            x_arr = np.linspace(r_start[0], r_end[0], num=n_rays_val)
            y_arr = np.linspace(r_start[1], r_end[1], num=n_rays_val)
        else:
            x_arr = np.linspace(xpos_val - beam_width_val / 2, xpos_val + beam_width_val / 2, num=n_rays_val)
            y_arr = np.linspace(ypos_val, ypos_val, num=n_rays_val)

    emitter_direction = np.array([0.0, 1.0])

    if n_angles_val <= 1:
        angle_arr = np.array([0.0])
    else:
        angle_arr = np.linspace(-beam_angle_val / 2, beam_angle_val / 2, n_angles_val)

    if n_colors_val <= 1:
        wavelength_arr = np.array([wavelength_val])
    else:
        wavelength_arr = np.linspace(440, 645, num=n_colors_val)

    line_idx = np.array([0], dtype=np.int32)

    # Trace all rays
    for r in range(x_arr.size):
        origin = np.array([x_arr[r], y_arr[r]])

        for c in range(n_colors_val):
            wavelength = wavelength_arr[c]
            n1 = get_ior(wavelength, medium_cauchy_a, medium_cauchy_b)
            n2 = get_ior(wavelength, sphere_cauchy_a, sphere_cauchy_b)

            for a in range(n_angles_val):
                angle = angle_arr[a]
                direction = rotate(emitter_direction, angle + rot_val)
                pol = 0.5 * intensity_val * np.array([(1 - polarization_val), (1 + polarization_val)])

                trace_single_ray(origin, direction, pol, wavelength, n1, n2,
                                 sphere_center, sphere_radius_val, depth_val, all_lines, line_idx)

    line_index = line_idx[0]

    # Create blank image for additive compositing
    img = np.zeros((img_height, img_width, 3), dtype=np.float32)

    # Render all lines to the image with additive blending
    render_lines_to_image(all_lines, line_index, img,
                          plot_xmin, plot_xmax, plot_ymin, plot_ymax,
                          scale, line_width_val)

    # Clip values to [0, 1] range
    img = np.clip(img, 0, 1)

    # Apply gamma correction for better visual appearance
    img = np.power(img, 1.0 / 2.2)

    # Clear and redraw
    ax.clear()
    ax.set_xlim([plot_xmin * scale, plot_xmax * scale])
    ax.set_ylim([plot_ymin * scale, plot_ymax * scale])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("rays: " + str(nrays))

    # Display the composited image
    ax.imshow(img, extent=(plot_xmin * scale, plot_xmax * scale, plot_ymin * scale, plot_ymax * scale),
              origin='upper', interpolation='bilinear', aspect='auto')

    # Plot bubble outline
    circle = plt.Circle(sphere_center, sphere_radius_val, fill=False, color=[1, 1, 1], linewidth=1)
    ax.add_patch(circle)

    return fig,

### SLIDERS ###

# Create the sliders
fig.subplots_adjust(left=0.5)

slider_x = 0.15
slider_y = 0.85
slider_space = 0.03
slider_width = 0.25

ax_nrays = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
nrays_slider = Slider(ax_nrays, "N Rays  ", valmin=1, valmax=max_rays, valinit=1, valstep=1, color="green")
nrays_slider.on_changed(update)
fig.add_axes(ax_nrays)

slider_y = slider_y - slider_space
ax_nangles = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
angles_slider = Slider(ax_nangles, "N Angles  ", valmin=1, valmax=max_angles, valinit=1, valstep=1, color="green")
angles_slider.on_changed(update)
fig.add_axes(ax_nangles)

slider_y = slider_y - slider_space
ax_ncolors = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
colors_slider = Slider(ax_ncolors, "N Colors  ", valmin=1, valmax=50, valinit=50, valstep=1, color="green")
colors_slider.on_changed(update)
fig.add_axes(ax_ncolors)

slider_y = slider_y - slider_space

slider_y = slider_y - slider_space
ax_width = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
beam_width_slider = Slider(ax_width, "Beam Width  ", valmin=0, valmax=6, valinit=2, valstep=0.01, color="green")
beam_width_slider.on_changed(update)
fig.add_axes(ax_width)

slider_y = slider_y - slider_space
ax_angle = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
beam_angle_val_slider = Slider(ax_angle, "Beam Angle  ", valmin=0, valmax=360, valinit=30, valstep=1, color="green")
beam_angle_val_slider.on_changed(update)
fig.add_axes(ax_angle)

slider_y = slider_y - slider_space
ax_rot = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
rot_slider = Slider(ax_rot, "Rotation  ", valmin=-180, valmax=180, valinit=0, color="green")
rot_slider.on_changed(update)
fig.add_axes(ax_rot)

slider_y = slider_y - slider_space
ax_xpos = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
xpos_slider = Slider(ax_xpos, "X Position  ", valmin=plot_xmin, valmax=plot_xmax, valinit=-0.9, color="green")
xpos_slider.on_changed(update)
fig.add_axes(ax_xpos)

slider_y = slider_y - slider_space
ax_ypos = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
ypos_slider = Slider(ax_ypos, "Y Position  ", valmin=plot_ymin, valmax=plot_ymax, valinit=-3, color="green")
ypos_slider.on_changed(update)
fig.add_axes(ax_ypos)

slider_y = slider_y - slider_space

slider_y = slider_y - slider_space
ax_sphere_xpos = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
sphere_xpos_slider = Slider(ax_sphere_xpos, "Sphere X Position  ", valmin=plot_xmin, valmax=plot_xmax, valinit=0,
                            color="green")
sphere_xpos_slider.on_changed(update)
fig.add_axes(ax_sphere_xpos)

slider_y = slider_y - slider_space
ax_sphere_ypos = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
sphere_ypos_slider = Slider(ax_sphere_ypos, "Sphere Y Position  ", valmin=plot_xmin, valmax=plot_xmax, valinit=0,
                            color="green")
sphere_ypos_slider.on_changed(update)
fig.add_axes(ax_sphere_ypos)

slider_y = slider_y - slider_space
ax_radius = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
sphere_radius_slider = Slider(ax_radius, "Sphere Radius ", valmin=0.1,
                              valmax=min(plot_xmax - plot_xmin, plot_ymax - plot_ymin) / 2, valinit=1, valstep=0.01,
                              color="green")
sphere_radius_slider.on_changed(update)
fig.add_axes(ax_radius)

slider_y = slider_y - slider_space

slider_y = slider_y - slider_space
ax_wavelength = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
wavelength_slider = Slider(ax_wavelength, "Wavelength  ", valmin=400, valmax=700, valinit=555, valstep=1, color="green")
wavelength_slider.on_changed(update)
fig.add_axes(ax_wavelength)

slider_y = slider_y - slider_space
ax_sphere_cauchy_a = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
sphere_cauchy_a_slider = Slider(ax_sphere_cauchy_a, "Sphere Cauchy A  ", valmin=1.0, valmax=2.0, valinit=1.33,
                                valstep=0.01, color="green")
sphere_cauchy_a_slider.on_changed(update)
fig.add_axes(ax_sphere_cauchy_a)

slider_y = slider_y - slider_space
ax_sphere_cauchy_b = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
sphere_cauchy_b_slider = Slider(ax_sphere_cauchy_b, "Sphere Cauchy B  ", valmin=0, valmax=0.02, valinit=0.0052,
                                valstep=0.0001, color="green")
sphere_cauchy_b_slider.on_changed(update)
fig.add_axes(ax_sphere_cauchy_b)

slider_y = slider_y - slider_space
ax_medium_cauchy_a = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
medium_cauchy_a_slider = Slider(ax_medium_cauchy_a, "Medium Cauchy A  ", valmin=1.0, valmax=2.0, valinit=1.0,
                                valstep=0.01, color="green")
medium_cauchy_a_slider.on_changed(update)
fig.add_axes(ax_medium_cauchy_a)

slider_y = slider_y - slider_space
ax_medium_cauchy_b = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
medium_cauchy_b_slider = Slider(ax_medium_cauchy_b, "Medium Cauchy B  ", valmin=0, valmax=0.02, valinit=0.0,
                                valstep=0.0001, color="green")
medium_cauchy_b_slider.on_changed(update)
fig.add_axes(ax_medium_cauchy_b)

slider_y = slider_y - slider_space
ax_polarization = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
polarization_slider = Slider(ax_polarization, "S/P Polarization  ", valmin=-1, valmax=1, valinit=0, valstep=0.01,
                             color="green")
polarization_slider.on_changed(update)
fig.add_axes(ax_polarization)

slider_y = slider_y - slider_space

slider_y = slider_y - slider_space
ax_brightness = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
brightness_slider = Slider(ax_brightness, "Brightness  ", valmin=0.0001, valmax=2, valinit=1, valstep=0.001,
                           color="green")
brightness_slider.on_changed(update)
fig.add_axes(ax_brightness)

slider_y = slider_y - slider_space
ax_linewidth = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
line_width_slider = Slider(ax_linewidth, "Line Width  ", valmin=0.1, valmax=10, valinit=3, valstep=0.01, color="green")
line_width_slider.on_changed(update)
fig.add_axes(ax_linewidth)

slider_y = slider_y - slider_space
ax_depth = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
depth_slider = Slider(ax_depth, "Depth  ", valmin=0, valmax=10, valinit=5, valstep=1, color="green")
depth_slider.on_changed(update)
fig.add_axes(ax_depth)

slider_y = slider_y - slider_space
ax_zoom = fig.add_axes([slider_x, slider_y, slider_width, 0.01])
zoom_slider = Slider(ax_zoom, "Zoom  ", valmin=0.1, valmax=10, valinit=1, color="green")
zoom_slider.on_changed(update)
fig.add_axes(ax_zoom)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('button_release_event', onrelease)
cid = fig.canvas.mpl_connect('motion_notify_event', onmove)
cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
cid = fig.canvas.mpl_connect('key_release_event', on_key_release)
cid = fig.canvas.mpl_connect('scroll_event', on_mouse_wheel)

figManager = plt.get_current_fig_manager()

ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

update(None)
plt.get_current_fig_manager().full_screen_toggle()
plt.show()