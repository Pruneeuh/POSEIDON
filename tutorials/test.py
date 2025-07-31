import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import streamlit as st
from io import BytesIO

img = mpimg.imread('LARD_example.jpg')
plt.imshow(img)

df = pd.read_csv('extract_labeling_Bing.csv', delimiter=';')
dic = df.to_dict(orient='records')[0]
TL = [dic['x_TL'],dic['y_TL']]
plt.scatter(TL[0], TL[1], label='Top Left')

TR = [dic['x_TR'],dic['y_TR']]
plt.scatter(TR[0], TR[1], label='Top Right')

BL = [dic['x_BL'],dic['y_BL']]
plt.scatter(BL[0], BL[1], label='Bottom Left')

BR = [dic['x_BR'],dic['y_BR']]
plt.scatter(BR[0], BR[1], label='Bottom Right')


plt.legend()
plt.show()
st.title("Affichage des points sur l'image")

# Affichage de l'image avec matplotlib dans un buffer
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(TL[0], TL[1], label='Top Left')
ax.scatter(TR[0], TR[1], label='Top Right')
ax.scatter(BL[0], BL[1], label='Bottom Left')
ax.scatter(BR[0], BR[1], label='Bottom Right')
ax.legend()

buf = BytesIO()
fig.savefig(buf, format="png")
plt.close(fig)
st.image(buf.getvalue(), caption="Image annotée", use_column_width=True)