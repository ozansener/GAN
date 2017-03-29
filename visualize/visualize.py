from __future__ import absolute_import, division, print_function, unicode_literals
import imageio
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw


def main():
  images_dir = '/Users/alan/Documents/research/nips2017/dcgan_results1'
  font_size = 30

  images = []
  font = ImageFont.truetype('/Library/Fonts/Arial.ttf', font_size)
  for file_name in sorted(os.listdir(images_dir)):
    if os.path.splitext(file_name)[-1] != '.png' or file_name == 'real_samples.png':
      continue

    image = Image.open(os.path.join(images_dir, file_name)).convert('RGBA')
    text = 'Epoch '+str(int(file_name.split('_')[-1].split('.')[0]))

    layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    w, h = draw.textsize(text, font=font)
    draw.text(((image.size[0]-w)//2, (image.size[1]-h)//2), text, font=font, fill=(255, 255, 255, 180))
    image = Image.alpha_composite(image, layer)

    images.append(image)
  images = np.stack(images)

  imageio.mimsave(os.path.join(images_dir, 'animation.gif'), images, duration=0.1)


if __name__ == '__main__':
  main()
