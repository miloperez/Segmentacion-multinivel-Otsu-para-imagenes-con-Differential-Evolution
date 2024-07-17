import ImageTools as IT
import Otsu as otsu

# file = "bojack.png"
file = "aegon277.png"

# mostrar imagen original en B&W
IT.displayBWImage(IT.loadImgBW(file), "Original")

K = 1       # número de cortes
prueba = otsu.Otsu(file, K)

# mostrar imagen resultante
IT.displayBWImage(prueba.getTransImg(), f"Trans Umb={K}")
