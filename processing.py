from rembg import remove
from PIL import Image
import numpy as np
import torch
import time
from scipy.ndimage import gaussian_filter
import os
import DracoPy

def processing(image_path, zoe):
    img = Image.open(image_path).convert("RGB")
   
    # Remove the background and
    extracted_np = remove(img, post_process_mask=True, only_mask=True)
    black_mask = np.invert(extracted_np)

    start_time = time.time()

    low_res_depth = zoe.infer_pil(img)
    low_res_scaled_depth = 2**16 - (low_res_depth - np.min(low_res_depth)) * 2**16 / (np.max(low_res_depth) - np.min(low_res_depth))
    # Image.fromarray((0.999 * low_res_scaled_depth).astype("uint16")).save('zoe_depth_map_16bit_low.png')

    im = np.asarray(img)
    tile_sizes = [[4, 4], [8, 8]]
    filters, compiled_tiles_list = [], []
    cos_pi = lambda val, dim: 0.998 * np.cos((abs(dim / 2 - val) / dim) * np.pi) ** 2

    for num_x, num_y in tile_sizes:
        M, N = im.shape[0] // num_x, im.shape[1] // num_y
        filter_dict = {key: np.zeros((M, N)) for key in [
            'right_filter', 'left_filter', 'top_filter', 'bottom_filter',
            'top_right_filter', 'top_left_filter', 'bottom_right_filter',
            'bottom_left_filter', 'filter']}

        for i in range(M):
            for j in range(N):
                x_val, y_val, xy_val = cos_pi(i, M), cos_pi(j, N), cos_pi(i, M) * cos_pi(j, N)
                filter_dict['right_filter'][i, j] = x_val if j > N / 2 else xy_val
                filter_dict['left_filter'][i, j] = x_val if j < N / 2 else xy_val
                filter_dict['top_filter'][i, j] = y_val if i < M / 2 else xy_val
                filter_dict['bottom_filter'][i, j] = y_val if i > M / 2 else xy_val
                for key, j_cond, i_cond in [
                    ('top_right_filter', j > N / 2, i < M / 2),
                    ('top_left_filter', j < N / 2, i < M / 2),
                    ('bottom_right_filter', j > N / 2, i > M / 2),
                    ('bottom_left_filter', j < N / 2, i > M / 2)]:
                    filter_dict[key][i, j] = 0.998 if j_cond and i_cond else (x_val if j_cond else (y_val if i_cond else xy_val))
                filter_dict['filter'][i, j] = xy_val

        filters.append(filter_dict)
        # for f in filter_dict:
        #     Image.fromarray((filter_dict[f] * 2**16).astype("uint16")).save(f'./outputs/mask_{f}_{num_x}_{num_y}.png')

    for i, (num_x, num_y) in enumerate(tile_sizes):
        M, N = im.shape[0] // num_x, im.shape[1] // num_y
        compiled_tiles = np.zeros(im.shape[:2])
        x_coords = list(range(0, im.shape[0], M))[:num_x]
        y_coords = list(range(0, im.shape[1], N))[:num_y]
        x_coords_all = x_coords + list(range(M // 2, im.shape[0], M))[:num_x - 1]
        y_coords_all = y_coords + list(range(N // 2, im.shape[1], N))[:num_y - 1]

        for x in x_coords_all:
            for y in y_coords_all:
                if np.all(black_mask[x:x+M, y:y+N] == 255): continue
                depth = zoe.infer_pil(Image.fromarray(np.uint8(im[x:x+M, y:y+N])))
                scaled_depth = 2**16 - (depth - np.min(depth)) * 2**16 / (np.max(depth) - np.min(depth))

                filter_keys = ['top_left_filter', 'top_filter','top_right_filter',
                            'left_filter', 'filter', 'right_filter',
                            'bottom_left_filter', 'bottom_filter', 'bottom_right_filter']
                
                x_index = 0 if x == min(x_coords_all) else 2 if x == max(x_coords_all) else 1
                y_index = 0 if y == min(y_coords_all) else 2 if y == max(y_coords_all) else 1

                selected_filter = filters[i][filter_keys[3*x_index + y_index]]

                compiled_tiles[x:x+M, y:y+N] += selected_filter * (
                    np.mean(low_res_scaled_depth[x:x+M, y:y+N]) +
                    np.std(low_res_scaled_depth[x:x+M, y:y+N]) *
                    ((scaled_depth - np.mean(scaled_depth)) / np.std(scaled_depth)))

        compiled_tiles[compiled_tiles < 0] = 0
        compiled_tiles_list.append(compiled_tiles)
        # Image.fromarray((2**16 * 0.999 * compiled_tiles / np.max(compiled_tiles)).astype("uint16")).save(f'tiled_depth_{i}.png')

    grey_im = np.mean(im,axis=2)

    tiles_diff = gaussian_filter(grey_im, sigma=20) - grey_im
    tiles_diff = np.clip(gaussian_filter(tiles_diff / np.max(tiles_diff), sigma=40) * 5, 0, 0.999)
    # Image.fromarray((tiles_diff*2**16).astype("uint16")).save('mask_image.png')

    combined_result = (tiles_diff * compiled_tiles_list[1] + (1-tiles_diff) * ((compiled_tiles_list[0] + low_res_scaled_depth) / 2)) / 2
    # Image.fromarray((2**16 * 0.999* combined_result / np.max(combined_result)).astype("uint16")).save('final_depth.png')

    print("--- %s seconds ---" % (time.time() - start_time))

    height_map = np.rot90(combined_result, -1)
    color_img = np.rot90(img, -1)
    mask = np.rot90(black_mask, -1)

    # Scale height values
    scaled_mesh = height_map.shape[0] * 0.3 * height_map / np.max(height_map)
    if len(scaled_mesh.shape) == 3:
        scaled_mesh = scaled_mesh[:, :, 0]

    # Generate vertices, colors, and faces
    vertices = []
    vertex_colors = []
    faces = []
    valid_vertex_map = {}

    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            if not mask[i, j]:
                vertices.append([i, j, scaled_mesh[i, j]])
                vertex_colors.append(color_img[i, j].tolist())
                valid_vertex_map[(i, j)] = len(vertices) - 1
                
    # ================== Save faces data for PLY format =================
    # for i in range(height_map.shape[0] - 1):
    #     for j in range(height_map.shape[1] - 1):
    #         if all((i+di, j+dj) in valid_vertex_map for di, dj in [(0,0), (0,1), (1,0), (1,1)]):
    #             v = [valid_vertex_map[(i+di, j+dj)] for di, dj in [(0,0), (0,1), (1,0), (1,1)]]
    #             faces.extend([[3, v[2], v[1], v[0]], [3, v[3], v[1], v[2]]])

    # data = []

    # #================= Write Point Cloud file (.xyz)=================
    # with open(f"{image_path[:-4]}.xyz", 'w') as f:
    #     for vertex, color in zip(vertices, vertex_colors):
    #         # scale color to 0-1
    #         color = [c / 255 for c in color]
    #         f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
    #         data.extend(vertex + color)

    # print(f"Point cloud saved to {image_path[:-4]}.xyz")


    # ================= Write .drc (Draco compressed) file =================
    try:
        vertices_np = np.array(vertices, dtype=np.float32)
        vertex_colors_np = np.array(vertex_colors, dtype=np.uint8)

        # Encode the point cloud
        compressed_data = DracoPy.encode(
            points=vertices_np.astype(np.float32),
            colors=vertex_colors_np.astype(np.uint8),  # Pass normalized colors
            compression_level=7  # Adjust compression level as needed
        )

        with open(f"{image_path[:-4]}.drc", 'wb') as f:
            f.write(compressed_data)

        print(f"Draco compressed point cloud saved to {image_path[:-4]}.drc")

    except Exception as e:
        print("Failed to generate Draco .drc file:", e)

    # #================= Write PLY file =================
    # with open("gata.ply", 'w') as f:
    #     f.write("ply\nformat ascii 1.0\n")
    #     f.write(f"element vertex {len(vertices)}\n")
    #     f.write("property float x\nproperty float y\nproperty float z\n")
    #     f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    #     f.write(f"element face {len(faces)}\n")
    #     f.write("property list uchar int vertex_indices\n")
    #     f.write("end_header\n")
        
    #     for vertex, color in zip(vertices, vertex_colors):
    #         f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        
    #     for face in faces:
    #         f.write(f"{' '.join(map(str, face))}\n")
    # print(f"PLY file saved to 'gata.ply'")
