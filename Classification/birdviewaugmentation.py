import cv2
import numpy as np
import os
import glob

# def birdview_augment(image, strength=0.3):
#     h, w = image.shape[:2]
#     dx = int(w * strength)
#     dy = int(h * strength)

#     src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
#     dst = np.float32([[dx, dy], [w - dx, dy], [w, h], [0, h]])

#     matrix = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(image, matrix, (w, h))
#     return warped

# def augment_and_save_to_new_folder(input_root, output_root, output_prefix="bird", strength=0.3):
#     os.makedirs(output_root, exist_ok=True)

#     for class_name in os.listdir(input_root):
#         input_class_path = os.path.join(input_root, class_name)
#         output_class_path = os.path.join(output_root, class_name)

#         if not os.path.isdir(input_class_path):
#             continue

#         os.makedirs(output_class_path, exist_ok=True)
#         image_paths = glob.glob(os.path.join(input_class_path, "*.jpg"))

#         for path in image_paths:
#             try:
#                 img = cv2.imread(path)
#                 warped = birdview_augment(img, strength=strength)

#                 base = os.path.basename(path)
#                 name, ext = os.path.splitext(base)
#                 out_path = os.path.join(output_class_path, f"{output_prefix}_{name}{ext}")
#                 cv2.imwrite(out_path, warped)
#                 print(f"✅ {out_path}")
#             except Exception as e:
#                 print(f"❌ Error on {path}: {e}")

# # Contoh pemakaian
# augment_and_save_to_new_folder(
#     input_root="dataset2/",
#     output_root="dataset3/",
#     strength=0.35
# )



# # import cv2
# # import numpy as np
# # import os
# # import glob

# # input_dir = 'dataset2/'
# # output_dir = 'dataset3/'
# # os.makedirs(output_dir, exist_ok=True)

# # # import glob


# # # for path in glob.glob(f'{input_dir}/*.jpg'):
# # #     img = cv2.imread(path)
# # #     warped = birdview_augment(img, strength=0.3)

# # #     filename = os.path.basename(path)
# # #     out_path = os.path.join(output_dir, f'bird_{filename}')
# # #     cv2.imwrite(out_path, warped)

# # def birdview_augment(image, strength=0.3):
# #     h, w = image.shape[:2]
# #     dx = int(w * strength)
# #     dy = int(h * strength)

# #     src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
# #     dst = np.float32([[dx, dy], [w - dx, dy], [w, h], [0, h]])

# #     matrix = cv2.getPerspectiveTransform(src, dst)
# #     warped = cv2.warpPerspective(image, matrix, (w, h))
# #     return warped

# # def augment_folder_birdview(root_dir, output_prefix="bird", strength=0.3):
# #     for class_name in os.listdir(root_dir):
# #         class_path = os.path.join(root_dir, class_name)
# #         if not os.path.isdir(class_path):
# #             continue

# #         image_paths = glob.glob(os.path.join(class_path, "*.jpg"))
# #         for path in image_paths:
# #             try:
# #                 img = cv2.imread(path)
# #                 warped = birdview_augment(img, strength=strength)

# #                 base = os.path.basename(path)
# #                 name, ext = os.path.splitext(base)
# #                 out_path = os.path.join(class_path, f"{output_prefix}_{name}{ext}")
# #                 cv2.imwrite(out_path, warped)
# #                 print(f"✅ {out_path}")
# #             except Exception as e:
# #                 print(f"❌ Error on {path}: {e}")


# # augment_folder_birdview("dataset/train", strength=0.35)

# import cv2
# import numpy as np
import random

# def warp_birdview_cctv(image, strength=0.3, skew=True):
#     h, w = image.shape[:2]
#     dx = int(w * strength)
#     dy = int(h * strength)

#     # randomisasi agar miring kiri/kanan
#     if skew:
#         skew_direction = random.choice(['left', 'right'])
#         if skew_direction == 'left':
#             dst = np.float32([
#                 [dx, dy],
#                 [w - dx//2, dy],
#                 [w, h],
#                 [0, h]
#             ])
#         else:
#             dst = np.float32([
#                 [dx//2, dy],
#                 [w - dx, dy],
#                 [w, h],
#                 [0, h]
#             ])
#     else:
#         dst = np.float32([
#             [dx, dy],
#             [w - dx, dy],
#             [w, h],
#             [0, h]
#         ])

#     src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
#     matrix = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(image, matrix, (w, h))
#     return warped

def motion_blur(image, size=7):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

def darken_image(image, factor=0.5):
    return np.clip(image * factor, 0, 255).astype('uint8')

# def simulate_cctv_view(image):
#     out = warp_birdview_cctv(image, strength=random.uniform(0.2, 0.4), skew=True)

#     # opsional: blur atau gelapkan
#     if random.random() < 0.6:
#         out = motion_blur(out, size=random.choice([5, 7]))

#     if random.random() < 0.4:
#         out = darken_image(out, factor=random.uniform(0.4, 0.7))

#     return out


# def augment_and_save_to_new_folder(input_root, output_root, output_prefix="bird", strength=0.3):
#     os.makedirs(output_root, exist_ok=True)

#     for class_name in os.listdir(input_root):
#         input_class_path = os.path.join(input_root, class_name)
#         output_class_path = os.path.join(output_root, class_name)

#         if not os.path.isdir(input_class_path):
#             continue

#         os.makedirs(output_class_path, exist_ok=True)
#         image_paths = glob.glob(os.path.join(input_class_path, "*.jpg"))

#         for path in image_paths:
#             try:
#                 img = cv2.imread(path)
#                 # warped = birdview_augment(img, strength=strength)
#                 warped = simulate_cctv_view(img)

#                 base = os.path.basename(path)
#                 name, ext = os.path.splitext(base)
#                 out_path = os.path.join(output_class_path, f"{output_prefix}_{name}{ext}")
#                 cv2.imwrite(out_path, warped)
#                 print(f"✅ {out_path}")
#             except Exception as e:
#                 print(f"❌ Error on {path}: {e}")

# # Contoh pemakaian
# augment_and_save_to_new_folder(
#     input_root="dataset2/",
#     output_root="dataset4/",
#     strength=0.35
# )


def realistic_cctv_style(img):
    h, w = img.shape[:2]
    
    # Simulasi miring dari atas (tanpa memperbesar ban)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle=random.uniform(-15, 15), scale=1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    # Sedikit shear (horizontal stretch untuk efek miring)
    pts1 = np.float32([[0,0], [w,0], [0,h]])
    shear = random.uniform(0.0, 0.3)
    pts2 = np.float32([[0,0], [w,0], [int(w*shear),h]])
    shear_mat = cv2.getAffineTransform(pts1, pts2)
    sheared = cv2.warpAffine(rotated, shear_mat, (w, h))

    # Optional: blur
    if random.random() < 0.6:
        sheared = motion_blur(sheared, size=random.choice([3,5,7]))

    # Optional: dark
    if random.random() < 0.4:
        sheared = darken_image(sheared, factor=random.uniform(0.4, 0.7))

    return sheared


def augment_and_save_to_new_folder(input_root, output_root, output_prefix="bird"):
    os.makedirs(output_root, exist_ok=True)

    for class_name in os.listdir(input_root):
        input_class_path = os.path.join(input_root, class_name)
        output_class_path = os.path.join(output_root, class_name)

        if not os.path.isdir(input_class_path):
            continue

        os.makedirs(output_class_path, exist_ok=True)
        image_paths = glob.glob(os.path.join(input_class_path, "*.jpg"))

        for path in image_paths:
            try:
                img = cv2.imread(path)
                # warped = birdview_augment(img, strength=strength)
                warped = realistic_cctv_style(img)

                base = os.path.basename(path)
                name, ext = os.path.splitext(base)
                out_path = os.path.join(output_class_path, f"{output_prefix}_{name}{ext}")
                cv2.imwrite(out_path, warped)
                print(f"✅ {out_path}")
            except Exception as e:
                print(f"❌ Error on {path}: {e}")

# Contoh pemakaian
augment_and_save_to_new_folder(
    input_root="dataset2/",
    output_root="dataset5ultra/",
)