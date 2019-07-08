from PIL import Image
grid = Image.open("pasted_grid2.png")
im = Image.open("pasted_grid2.png")
print(im)
region = im.crop((391, 401, 438, 477))

print(region)
grid.paste(region, (491, 401, 538, 477))
print(grid)
grid.save("pasted_grid.png")