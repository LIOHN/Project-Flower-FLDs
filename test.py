import matplotlib

image = mpimg.imread("img_00000001.jpg")
points = np.array([[330,620],[950,620],[692,450],[587,450]])

plt.imshow(image)
plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="red", s=200)
plt.show()