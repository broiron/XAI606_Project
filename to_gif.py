import os
import imageio
from argparse import ArgumentParser
from pathlib import Path
# Get a list of all png files in the current directory


parser = ArgumentParser(description="output gif")
parser.add_argument("--render_path", '-r', required=True, nargs="+", type=str)

args = parser.parse_args()

dir_path = Path(args.render_path[0])

print(dir_path)

png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
png_files.sort()

# Read each image file and append it to a list
images = []
for file in png_files:
    images.append(imageio.imread(os.path.join(dir_path, file)))

# Save the images as a gif
imageio.mimsave(dir_path / 'output.gif', images)

