from PIL import Image
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x * ratio) for x in size])
    im = im.resize(new_image_size, 3)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
    return new_im

def clean_image_data(input_folder, output_folder, final_size=512):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    file_list = os.listdir(input_folder)

    for n, item in enumerate(file_list, 1):
        # Open the image file
        im = Image.open(os.path.join(input_folder, item))

        # Resize the image and save it in the output folder
        new_im = resize_image(final_size, im)
        new_im.save(os.path.join(output_folder, item))
        print(item)

if __name__ == '__main__':
    input_path = ""  # input image folder path
    output_path = ""  # Input the path for where cleaned images will be saved
    final_image_size = 512

    clean_image_data(input_path, output_path, final_image_size)
