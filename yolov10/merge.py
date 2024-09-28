from PIL import Image



if __name__ == '__main__':
    # 打开四张图片
    img1 = Image.open('result0.jpg')
    img2 = Image.open('result1.jpg')
    img3 = Image.open('result2.jpg')
    img4 = Image.open('result3.jpg')

    # 假设所有图片的尺寸相同
    width, height = img1.size

    # 创建一个新的空白图像，用来放置四张图片
    combined_image = Image.new('RGB', (4 * width, height))

    # 粘贴四张图片到新图像上
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (width, 0))
    combined_image.paste(img3, (2 * width, 0))
    combined_image.paste(img4, (3 * width, 0))

    # 保存合成后的图像
    combined_image.save('v10_self_val_merge.jpg')

