import cv2
#读取双目相机拍摄的整体图像：
full_image=cv2.imread('ping_ban.png')
#获取图像的高度和宽度
height,width=full_image.shape[:2]
print('图片高度：',height)
print('width:',width)

#分割为左眼视图和右眼视图
left_image=full_image[:,0:width//2]#左边部分 ,第三个维度是图像的通道数（RGB为3通道）
right_image=full_image[:,width//2:width]#右半部分
#显示分割后的左右眼视图
cv2.imshow('left image',left_image)
cv2.imshow('right image',right_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('ping_ban_l.jpg',left_image)
cv2.imwrite('ping_ban_r.jpg',right_image)
