from skimage import io, color, feature, morphology, exposure, measure, filters
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk, diamond, square, ball  # noqa
from skimage.segmentation import chan_vese
from skimage import feature
from skimage import data, img_as_float
from skimage import data
from skimage import transform
from skimage.feature import CENSURE
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.segmentation import flood, flood_fill
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage import exposure


def show_images(image, result_image):

    # Display both the original and final images side by side
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(121)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original Image')
    plt.axis('off')

    # Final image with only darkest pixels retained
    plt.subplot(122)
    plt.imshow(result_image, cmap=plt.cm.gray)
    plt.title('Darkest Pixels Retained, Others Turned White')
    plt.axis('off')

    plt.tight_layout()
    # plt.show()


def erosion_(image):
    footprint = disk(1)  # square etc...
    eroded = erosion(image, footprint)
    return eroded


def dilation_(image):
    footprint = disk(1)  # square etc...
    dilated = dilation(image, footprint)
    return dilated


def opening_(image):
    footprint = disk(1)  # square etc...
    opened = opening(image, footprint)
    return opened


def closing_(image):
    footprint = disk(1)  # square etc...
    closed = closing(image, footprint)
    return closed


def black_tophat_(image):
    footprint = diamond(3)  # square etc...
    b_tophat = black_tophat(image, footprint)
    return b_tophat


def chan_vese_(image):
    cv = chan_vese(image, mu=0.5, lambda1=1, lambda2=1, tol=1e-3,
                   max_num_iter=30, dt=0.5, init_level_set="checkerboard",
                   extended_output=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(cv[0], cmap="gray")
    ax[1].set_axis_off()
    title = f'Chan-Vese segmentation - {len(cv[2])} iterations'
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(cv[1], cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Final Level Set", fontsize=12)

    ax[3].plot(cv[2])
    ax[3].set_title("Evolution of energy over iterations", fontsize=12)

    fig.tight_layout()
    # plt.show()


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def equalize(img):
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)  # 0.03

    return img_adapteq

def corrections(img):

    gamma_corrected = exposure.adjust_gamma(img, 0.4)  # Gamma
    logarithmic_corrected = exposure.adjust_log(img, 1)  # Logarithmic

    return logarithmic_corrected


def canny_(image):
    edges = feature.canny(image, sigma=3)
    return edges


def sobel_edge_detector(image):
    edge_sobel = filters.roberts(image)
    return edge_sobel


def find_contour(image):
    # Convert the image to grayscale if it's in color
    if image.shape[-1] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image

    # Find contours in the grayscale image
    contours = measure.find_contours(gray_image, 0.5)  # Adjust the threshold as needed

    # Display the original image
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot the detected contours
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, c='r')

    # plt.show()
    return contours


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def morph_acwe(image):
    # Morphological ACWE

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                 smoothing=3, iter_callback=callback)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 35")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)


    # Morphological GAC
    image = img_as_float(data.coins())
    gimage = inverse_gaussian_gradient(image)

    # Initial level set
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, num_iter=230,
                                               init_level_set=init_ls,
                                               smoothing=1, balloon=-1,
                                               threshold=0.69,
                                               iter_callback=callback)

    ax[2].imshow(image, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("Morphological GAC segmentation", fontsize=12)

    ax[3].imshow(ls, cmap="gray")
    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 230")
    ax[3].legend(loc="upper right")
    title = "Morphological GAC evolution"
    ax[3].set_title(title, fontsize=12)

    fig.tight_layout()
    # plt.show()


def find_corner(image):
    coords = corner_peaks(corner_harris(image), min_distance=1, threshold_rel=0.01)  # min_distance=5, threshold_rel=0.02
    coords_subpix = corner_subpix(image, coords, window_size=13)  # window_size= 13

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            linestyle='None', markersize=6)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    ax.axis((0, 310, 200, 0))
    plt.show()

    return coords


def censure(image):
    tform = transform.AffineTransform(scale=(1.5, 1.5), rotation=0.1,
                                      translation=(150, -200))
    img_warp = transform.warp(image, tform)

    detector = CENSURE()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    detector.detect(image)

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[0].set_title("Original Image")

    detector.detect(img_warp)

    ax[1].imshow(img_warp, cmap=plt.cm.gray)
    ax[1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[1].set_title('Transformed Image')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    # plt.show()

    return detector.keypoints


def flood_fill_(image, pos):
    pairs_array = np.array(pos)

    # Calculate the mean pair
    mean_pair = np.mean(pairs_array, axis=0)
    mean_pair = np.round(mean_pair).astype(int)

    # light_coat = flood_fill(image, tuple(pos[0]), 255, tolerance=0.4)
    # light_coat = flood_fill(image, tuple(pos[0]), 255, tolerance=0.1)
    light_coat = flood_fill(image, tuple(mean_pair), 255, tolerance=0.1)
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(light_coat, cmap=plt.cm.gray)
    ax[1].plot(tuple(mean_pair)[0], tuple(mean_pair)[1], 'ro')  # seed point
    ax[1].set_title('After flood fill')
    ax[1].axis('off')

    # plt.show()


def angle_calc(polygon):
    """
       Calculate the smallest angle in a polygon.

       Parameters:
       polygon (list of tuples): List of (x, y) coordinates representing the polygon vertices.

       Returns:
       float: The smallest angle in degrees.
       """
    if len(polygon) < 3:
        return None  # A polygon must have at least 3 vertices

    smallest_angle = float('inf')

    for i in range(len(polygon)):
        # Get three consecutive vertices
        prev_vertex = polygon[i - 1]
        current_vertex = polygon[i]
        next_vertex = polygon[(i + 1) % len(polygon)]  # Wrap around for the last vertex

        # Calculate vectors from the current vertex to the previous and next vertices
        vector1 = np.array(prev_vertex) - np.array(current_vertex)
        vector2 = np.array(next_vertex) - np.array(current_vertex)

        # Calculate the angle between the two vectors using the dot product
        angle_rad = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        angle_deg = np.degrees(angle_rad)

        # Update the smallest angle if necessary
        if angle_deg < smallest_angle:
            smallest_angle = angle_deg

    return smallest_angle


def calculate_total_polygon_angle(polygon):
    """
    Calculate the total angle in degrees of a polygon.

    Parameters:
    polygon (list of tuples): List of (x, y) coordinates representing the polygon vertices.

    Returns:
    float: The total angle in degrees.
    """
    if len(polygon) < 3:
        return None  # A polygon must have at least 3 vertices

    total_angle = 0.0

    for i in range(len(polygon)):
        # Get three consecutive vertices
        prev_vertex = polygon[i - 1]
        current_vertex = polygon[i]
        next_vertex = polygon[(i + 1) % len(polygon)]  # Wrap around for the last vertex

        # Calculate vectors from the current vertex to the previous and next vertices
        vector1 = np.array(prev_vertex) - np.array(current_vertex)
        vector2 = np.array(next_vertex) - np.array(current_vertex)

        # Calculate the angle between the two vectors using the dot product
        angle_rad = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        angle_deg = np.degrees(angle_rad)

        # Add the angle to the total
        total_angle += angle_deg

    return total_angle


def create_polygon(pos):
    # Calculate the centroid of the coordinates
    centroid = np.mean(pos, axis=0)

    # Sort the coordinates based on their polar angles relative to the centroid
    sorted_coordinates = sorted(pos,
                                key=lambda coord: np.arctan2(coord[1] - centroid[1], coord[0] - centroid[0]))

    # Extract x and y coordinates from the sorted list
    x_coords, y_coords = zip(*sorted_coordinates)

    # Rotate the polygon 90 degrees to the right
    rotated_x_coords = [centroid[0] - (y - centroid[1]) for x, y in sorted_coordinates]
    rotated_y_coords = [centroid[1] + (x - centroid[0]) for x, y in sorted_coordinates]

    # Create a 256x256 canvas
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Plot the rotated coordinates as a polygon
    plt.plot(rotated_x_coords + [rotated_x_coords[0]], rotated_y_coords + [rotated_y_coords[0]], 'r')

    # Display the plot
    plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
    plt.gca().invert_xaxis()  # Invert the y-axis to match image coordinates
    plt.axis('off')  # Turn off axis labels and ticks
    # plt.show()

    rotated_polygon = [(x, y) for x, y in zip(rotated_x_coords, rotated_y_coords)]
    return rotated_polygon


def create_triangle(pos):
    # Calculate the centroid of the coordinates

    centroid = np.mean(pos, axis=0)

    # Rotate the original coordinates 90 degrees to the right
    pos = [(centroid[0] + (y - centroid[1]), centroid[1] - (x - centroid[0])) for x, y in pos]

    # Sort the coordinates based on their polar angles relative to the centroid
    sorted_coordinates = sorted(pos,
                                key=lambda coord: np.arctan2(coord[1] - centroid[1], coord[0] - centroid[0]))

    # Extract x and y coordinates from the sorted list
    x_coords, y_coords = zip(*sorted_coordinates)

    # Add the first coordinate at the end to close the polygon
    x_coords = list(x_coords) + [x_coords[0]]
    y_coords = list(y_coords) + [y_coords[0]]

    # # Rotate the polygon 90 degrees to the right
    # rotated_x_coords = [centroid[0] - (y - centroid[1]) for x, y in sorted_coordinates]
    # rotated_y_coords = [centroid[1] + (x - centroid[0]) for x, y in sorted_coordinates]
    # Rotate the polygon 90 degrees to the right


    # Calculate the lengths of edges
    edge_lengths = [np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for (x1, y1), (x2, y2) in
                    zip(sorted_coordinates, sorted_coordinates[1:] + [sorted_coordinates[0]])]

    # Find the indices of the two longest edges
    indices_of_longest_edges = np.argsort(edge_lengths)[-2:]

    # Extract the coordinates of the two longest edges
    longest_edge1 = [sorted_coordinates[indices_of_longest_edges[0]],
                     sorted_coordinates[(indices_of_longest_edges[0] + 1) % len(pos)]]
    longest_edge2 = [sorted_coordinates[indices_of_longest_edges[1]],
                     sorted_coordinates[(indices_of_longest_edges[1] + 1) % len(pos)]]

    # Create a 256x256 canvas
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Plot the sorted coordinates as a polygon
    plt.plot(x_coords, y_coords, 'r')

    # Plot the two longest edges as a triangle
    triangle_x = [longest_edge1[0][0], longest_edge1[1][0], longest_edge2[1][0]]
    triangle_y = [longest_edge1[0][1], longest_edge1[1][1], longest_edge2[1][1]]
    plt.fill(triangle_x, triangle_y, 'b', alpha=0.5, label='Triangle')

    # Display the legend
    plt.legend()

    # Display the plot
    # plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
    plt.gca().invert_xaxis()  # Invert the y-axis to match image coordinates
    plt.axis('on')  # Turn off axis labels and ticks
    # plt.show()


def calculate_intersection(edge1, edge2):
    """
    Calculate the intersection point of two infinitely extended line segments.

    Parameters:
    edge1 (tuple): Tuple of two points representing the first edge.
    edge2 (tuple): Tuple of two points representing the second edge.

    Returns:
    tuple or None: The intersection point as a tuple (x, y), or None if the lines are parallel.
    """
    x1, y1 = edge1[0]
    x2, y2 = edge1[1]
    x3, y3 = edge2[0]
    x4, y4 = edge2[1]

    # Calculate determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if det == 0:
        return None  # Lines are parallel, no intersection

    # Calculate intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return px, py


def create_new_polygon(polygon):
    """
    Add an intersection point to the two longest edges of a polygon (infinite length) and plot it.

    Parameters:
    polygon (list of tuples): List of (x, y) coordinates representing the polygon vertices.

    Returns:
    list of tuples: A new polygon with the intersection point.
    """

    if len(polygon) < 3:
        return None  # A polygon must have at least 3 vertices

    # Calculate edge lengths and store them with their corresponding vertices
    edges = []
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]  # Wrap around for the last vertex
        edge_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        edges.append(((x1, y1), (x2, y2), edge_length))

    # Sort the edges by length in descending order
    edges.sort(key=lambda edge: -edge[2])

    # Extract the two longest edges
    longest_edge1 = edges[0]
    longest_edge2 = edges[1]

    # Calculate the intersection point (infinite length)
    intersection = calculate_intersection(longest_edge1, longest_edge2)

    if intersection is None:
        return None  # Lines are parallel, no intersection point

    # Find the index of the first longest edge in the original polygon
    index_longest_edge1 = polygon.index(longest_edge1[0])

    # Create a new polygon by adding the intersection point after the first longest edge
    new_polygon = polygon[:index_longest_edge1 + 1] + [intersection] + polygon[index_longest_edge1 + 2:]

    # Plot the new polygon
    x_coords, y_coords = zip(*new_polygon)

    # Create a 256x256 canvas
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Plot the new polygon
    plt.plot(x_coords + (x_coords[0],), y_coords + (y_coords[0],), 'b', label='New Polygon')

    # Display the legend
    plt.legend()

    # Display the plot
    plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
    plt.axis('off')  # Turn off axis labels and ticks
    # plt.show()

    return new_polygon


def calculate_polygon_area(polygon):
    """
    Calculate the area of a polygon using the shoelace formula.

    Parameters:
    polygon (list of tuples): List of (x, y) coordinates representing the polygon vertices.

    Returns:
    float: The area of the polygon.
    """
    if len(polygon) < 3:
        return 0.0  # A polygon with less than 3 vertices has zero area

    x_coords, y_coords = zip(*polygon)
    x_coords = list(x_coords)
    y_coords = list(y_coords)

    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    area = 0.0
    for i in range(len(polygon)):
        area += x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]

    area = 0.5 * abs(area)
    return area


def show_and_compare_polygons(poly1, poly2):
    """
    Display two polygons on a 256x256 canvas and print the difference in area between them.

    Parameters:
    poly1 (list of tuples): List of (x, y) coordinates representing the first polygon vertices.
    poly2 (list of tuples): List of (x, y) coordinates representing the second polygon vertices.
    """
    # Create a 256x256 canvas
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Plot the first polygon
    x_coords1, y_coords1 = zip(*poly1)
    x_coords1 += (x_coords1[0],)
    y_coords1 += (y_coords1[0],)
    plt.plot(x_coords1, y_coords1, 'r', label='Polygon 1')

    # Plot the second polygon
    x_coords2, y_coords2 = zip(*poly2)
    x_coords2 += (x_coords2[0],)
    y_coords2 += (y_coords2[0],)
    plt.plot(x_coords2, y_coords2, 'g', label='Polygon 2')

    # Display the legend
    plt.legend()

    # Calculate and print the difference in area between the two polygons
    area1 = calculate_polygon_area(poly1)
    area2 = calculate_polygon_area(poly2)
    area_difference = abs(area1 - area2)
    print(f"Difference in area: {area_difference:.2f} square units")

    # Display the plot
    plt.gca().invert_yaxis()  # Invert the y-axis to match image coordinates
    plt.gca().invert_xaxis()  # Invert the y-axis to match image coordinates
    plt.axis('off')  # Turn off axis labels and ticks
    # plt.show()

    return area_difference


def return_params(image_path):
    # Load the image
    image = io.imread(image_path)
    image = color.rgb2gray(image)

    # image = equalize(image)
    logarithmic_corrected = corrections(image)
    result_image = erosion_(logarithmic_corrected)
    result_image = dilation_(result_image)
    # result_image = opening_(result_image)
    result_image = closing_(result_image)

    pos = find_corner(result_image)  # todo: continue

    # Filter pairs based on the threshold
    pos = [pair for pair in pos if pair[0] >= 20 and pair[1] >= 20]

    # Filter pairs based on the threshold
    pos = [pair for pair in pos if pair[0] <= 230 and pair[1] <= 230]  # todo : work with polygon

    polygon = create_polygon(pos)

    smallest_angle = angle_calc(polygon)
    print(f"The smallest angle in the polygon is {smallest_angle:.2f} degrees.")
    total_angle = calculate_total_polygon_angle(polygon)
    print(f"The total angle in the polygon is {total_angle:.2f} degrees.")
    # create_triangle(polygon)
    new_polygon = create_new_polygon(polygon)
    area_diff = show_and_compare_polygons(polygon, new_polygon)

    # show_images(image, result_image)

    return smallest_angle, total_angle, area_diff


def main(image_path):
    # Load the image
    image = io.imread(image_path)
    image = color.rgb2gray(image)

    # image = equalize(image)
    logarithmic_corrected = corrections(image)

    result_image = erosion_(logarithmic_corrected)
    result_image = dilation_(result_image)
    # result_image = opening_(result_image)
    result_image = closing_(result_image)

    pos = find_corner(result_image)  # todo: continue

    # Filter pairs based on the threshold
    pos = [pair for pair in pos if pair[0] >= 20 and pair[1] >= 20]

    # Filter pairs based on the threshold
    pos = [pair for pair in pos if pair[0] <= 230 and pair[1] <= 230]  #  todo : work with polygon

    polygon = create_polygon(pos)

    smallest_angle = angle_calc(polygon)
    print(f"The smallest angle in the polygon is {smallest_angle:.2f} degrees.")
    total_angle = calculate_total_polygon_angle(polygon)
    print(f"The total angle in the polygon is {total_angle:.2f} degrees.")
    # create_triangle(polygon)
    new_polygon = create_new_polygon(polygon)
    area_diff = show_and_compare_polygons(polygon, new_polygon)

    show_images(image, result_image)

    return smallest_angle, total_angle, area_diff


if __name__ == "__main__":
   image_path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\ק\עיבוד תמונה\Lapis\256\good\10.jpg"

   a, b, c = main(image_path)