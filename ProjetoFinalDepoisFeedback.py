from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np

Imagem = cv2.imread("ImagensDoProjeto/diaretdb1_image053.png")

# Diminuindo tamanho da imagem
scalaPercent = 20
widht = int(Imagem.shape[1] * scalaPercent / 100)
height = int(Imagem.shape[0] * scalaPercent / 100)
dim = (widht, height)
Imagem1 = cv2.resize(Imagem, dim, interpolation=cv2.INTER_AREA)

# Aplicando filtro para eliminar ruido
Imagem2 = cv2.GaussianBlur(src=Imagem1, ksize=(3,3), sigmaX=0)

# Extracao do canal verde
Imagem3 = Imagem1.copy()
Imagem3[:,:,0] = 0
Imagem3[:,:,2] = 0

# Conversoes em tons de cinza
Imagem4 = cv2.cvtColor(Imagem1, cv2.COLOR_BGR2GRAY)
Imagem4 = cv2.cvtColor(Imagem4, cv2.COLOR_GRAY2BGR)


# Contatenado imagens e exibindo
final = np.concatenate((Imagem1, Imagem2), axis=1)
final2 = np.concatenate((Imagem3, Imagem4), axis=1)

im3 = Imagem1.copy()


g = im3[:,:,1]
alpha = 1.5
beta = 10
out = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)
out2 = out.copy()
edges = cv2.Canny(out,20,120)
#cv2.imshow("Canny", edges)


cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(out, cnts, [255,255,255])
#cv2.imshow("Preenchimento", out)


rs = out - edges
_, rs2 = cv2.threshold(rs, 130, 255, cv2.THRESH_BINARY)
#cv2.imshow("Binarizacao", rs2)


image = rs2.copy()
kernel = np.ones((5, 5), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)


'''
plt.subplot(121),plt.imshow(rs2, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''


img = image.copy()
selected_contours = []
imagens = []


for contour in contours:
  area = cv2.contourArea(contour)
  if area < 2:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)

m2 = skimage.color.gray2rgb(imagens[-1])
rss2 = cv2.addWeighted(im3, 1, m2, 0.7, 0)
final3 = np.concatenate((m2, rss2), axis=1)
finalR = np.concatenate((final, final2, final3), axis=0)

# Colocando texto nas imagens e exibindo
font = cv2.FONT_HERSHEY_PLAIN
org = (50, 50)
fontScale = 1
color = (255, 255, 0)
thickness = 2

cv2.putText(finalR, "Imagem Original", org, font, fontScale, color, thickness)
cv2.putText(finalR, "Filtro Gaussiano", (Imagem2.shape[1]+50, 50), font, fontScale, color, thickness)
cv2.putText(finalR, "Canal Verde", (50, Imagem3.shape[1]), font, fontScale, color, thickness)
cv2.putText(finalR, "Tons de Cinza", (Imagem4.shape[1]+50, Imagem4.shape[1]), font, fontScale, color, thickness)
cv2.putText(finalR, "Microaneurismo", (50, m2.shape[1]+220), font, fontScale, color, thickness)
cv2.putText(finalR, "Imagem Final", (m2.shape[1]+50, m2.shape[1]+220), font, fontScale, color, thickness)


cv2.imshow("teste", finalR)
