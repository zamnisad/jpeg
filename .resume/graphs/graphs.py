from Main.Main import *
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def generate_image_versions(input_path):
    img = Image.open(input_path)
    
    output_dir = os.path.dirname(input_path)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    gray_img = img.convert("L")
    gray_path = os.path.join(output_dir, f"{filename}_grayscale.png")
    gray_img.save(gray_path)
    print(f"Saved grayscale: {gray_path}")
    
    bw_img = gray_img.convert("1")
    bw_path = os.path.join(output_dir, f"{filename}_bw_no_dither.png")
    bw_img.save(bw_path)
    print(f"Saved BW without dithering: {bw_path}")

    dithered_img = gray_img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
    dithered_path = os.path.join(output_dir, f"{filename}_bw_dithered.png")
    dithered_img.save(dithered_path)
    print(f"Saved BW with dithering: {dithered_path}")


def two298(dir: str):
    comp_dir = os.path.join(dir, 'compressed')
    decomp_dir = os.path.join(dir, 'decompressed')
    
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(decomp_dir, exist_ok=True)
    
    for img_name in os.listdir(dir):
        img_path = os.path.join(dir, img_name)
        
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_name)[0]

            img_comp_dir = os.path.join(comp_dir, base_name)
            img_decomp_dir = os.path.join(decomp_dir, base_name)
            os.makedirs(img_comp_dir, exist_ok=True)
            os.makedirs(img_decomp_dir, exist_ok=True)
            
            sizes = []
            qualities = list(range(2, 100, 2))
            
            for q in tqdm(qualities, f'Processing {base_name}'):
                e_zmn_name = f"e_{base_name}_{q}.zmn"
                e_zmn_path = os.path.join(img_comp_dir, e_zmn_name)
                
                d_zmn_name = f"d_{base_name}_{q}.png"
                d_zmn_path = os.path.join(img_decomp_dir, d_zmn_name)

                compressor = Main(e_zmn_path)
                compressor.encode(img_path, quality=q)

                file_size = os.path.getsize(e_zmn_path)
                sizes.append(file_size / 1024)
                
                # print(e_zmn_path, d_zmn_path)

                compressor.decode(d_zmn_path)



def plot_compression_sizes(
    base_path="Z:/prog/jpeg/.resume/test_imgs",
    output_path="Z:/prog/jpeg/.resume/imgs"
):
    os.makedirs(output_path, exist_ok=True)

    raw_path = os.path.join(base_path)
    compressed_base = os.path.join(base_path, "compressed")

    for file in os.listdir(raw_path):
        if not file.endswith('.raw'):
            continue

        name = os.path.splitext(file)[0]
        compressed_folder = os.path.join(compressed_base, name)

        sizes = []
        qualities = []

        for q in range(2, 100, 2):
            fname = f"e_{name}_{q}.zmn"
            fpath = os.path.join(compressed_folder, fname)
            if os.path.isfile(fpath):
                sizes.append(os.path.getsize(fpath) / 1024)  # размер в КБ
                qualities.append(q)
            else:
                print(f"[!] Missing: {fpath}")

        if not sizes:
            print(f"[!] No compressed files found for {name}")
            continue

        # Строим график
        plt.figure(figsize=(10, 6))
        plt.plot(qualities, sizes, marker='o')
        plt.title(f"Размер файла vs Качество JPEG для {name}")
        plt.xlabel("Качество JPEG")
        plt.ylabel("Размер файла (КБ)")
        plt.grid(True)
        plt.tight_layout()

        # Сохраняем
        out_path = os.path.join(output_path, f"{name}_compression_plot.png")
        plt.savefig(out_path)
        plt.close()

        print(f"[ok] Saved plot for {name} -> {out_path}")


def zero2hundred(dir: str):
    comp_dir = dir
    decomp_dir = dir
    
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(decomp_dir, exist_ok=True)
    
    for img_name in os.listdir(dir):
        img_path = os.path.join(dir, img_name)
        
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_name)[0]
            
            sizes = []
            qualities = list(range(0, 101, 20))
            
            for q in tqdm(qualities, f'Processing {base_name}'):
                e_zmn_name = f"{base_name}_{q}.zmn"
                e_zmn_path = os.path.join(comp_dir, e_zmn_name)
                
                d_zmn_name = f"{base_name}_{q}.png"
                d_zmn_path = os.path.join(decomp_dir, d_zmn_name)

                compressor = Main(e_zmn_path)
                compressor.encode(img_path, quality=q)

                file_size = os.path.getsize(e_zmn_path)
                sizes.append(file_size / 1024)
                
                # print(e_zmn_path, d_zmn_path)

                compressor.decode(d_zmn_path)
                os.remove(e_zmn_path)
                

def make_multiple_collages(folder: str, out_dir: str, figsize=(12, 8)) -> None:
    """
    Строит 2x3 коллажи для каждого набора изображений NAME_Q.png с Q ∈ [0, 20, ..., 100].

    Args:
        folder (str): Путь к папке, содержащей изображения.
        out_dir (str): Куда сохранять коллажи.
        figsize (tuple): Размер коллажа (ширина, высота).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Сбор изображений по NAME
    name_to_files = defaultdict(list)
    for f in os.listdir(folder):
        if not f.lower().endswith('.png'):
            continue
        if '_' not in f:
            continue
        name, q_ext = f.rsplit('_', 1)
        q = q_ext.split('.')[0]
        try:
            q = int(q)
            name_to_files[name].append((q, os.path.join(folder, f)))
        except ValueError:
            continue  # Если Q не число — пропускаем

    for name, items in name_to_files.items():
        # Сортируем по значению качества
        items_sorted = sorted(items, key=lambda x: x[0])
        if len(items_sorted) != 6:
            print(f"[!] Пропущено {name} — найдено {len(items_sorted)} изображений, ожидалось 6")
            continue

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        for ax, (q, path) in zip(axes.flat, items_sorted):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(f"{name}, Q={q}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"collage_{name}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[ok] Сохранён коллаж: {out_path}")



if __name__ == "__main__":
    test_dir = r"Z:\prog\jpeg\.resume\test_imgs\from0to100"

    for img_file in ['Lenna.png', 'Big.png']:
        input_image = os.path.join(test_dir, img_file)
        generate_image_versions(input_image)

    two298(test_dir)
    plot_compression_sizes()
    zero2hundred(test_dir)
    make_multiple_collages(test_dir, r'Z:\prog\jpeg\.resume\imgs')