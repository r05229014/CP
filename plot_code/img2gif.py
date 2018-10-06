import imageio
import os 

def img2gif(dir, case):
    images = []
    png_dir = dir + case + '/'
    print(sorted(os.listdir(png_dir)))
    for filename in sorted(os.listdir(png_dir)):
        if filename.endswith('.png'):
            file_path = os.path.join(png_dir, filename)
            images.append(imageio.imread(file_path))
    imageio.mimsave('./gif/Linear/' + case + '.gif', images)

png_dir = '../img/paper/Linear/'
name_list = os.listdir(png_dir)
for name in name_list:
    print('Now ploting : %s' %name_list)
    img2gif(png_dir, name)
