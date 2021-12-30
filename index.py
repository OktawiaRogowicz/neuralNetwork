import streamlit as st
import os
import json
import altair as alt
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import threading
from threading import current_thread

# import the time module
import time
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server

TIME = 10
timer = TIME


def countdown(counter):
    global timer
    while timer >= 0:
        counter.markdown("## %d" % timer)
        time.sleep(1)
        timer -= 1
        counter.empty()


def trim():
    n = 1
    nfirstlines = []

    with open("categories.txt") as f, open("categoriestmp.txt", "w") as out:
        for x in range(n):
            try:
                nfirstlines.append(next(f))
            except StopIteration:
                return []
        for line in f:
            out.write(line)
    f.close()
    out.close()
    os.remove("categories.txt")
    os.rename("categoriestmp.txt", "categories.txt")
    return nfirstlines


def main():
    st.title("")
    counter = st.empty()

    category = trim()
    print(category)
    if len(category) == 0:
        # game finished
        categories = ['cookie', 'crab', 'carrot', 'bat', 'floor lamp', 'grass', 'moon', 'mug', 'sword', 'sun']
        f = open("categories.txt", "w")
        for c in categories:
            f.write(c + "\n")
        f.close()
    st.header(category[0])

    bg_color = "#ffffff"

    global timer
    if timer == 0:
        if st.button("Next"):
            bg_color = "#ffffff"

    canvas_result = st_canvas(
        fill_color="#000000",  # Fixed fill color with some opacity
        stroke_width=5,
        stroke_color='#000000',
        background_color=bg_color,
        update_streamlit=False,
        height=512, width=512,
        drawing_mode='freedraw',
        display_toolbar=True,
        key='canvas'
    )

    countdown(counter)

    pic = Image.fromarray(canvas_result.image_data, 'RGBA').convert('L')
    pic.thumbnail((128, 128), Image.ANTIALIAS)
    arr = np.asarray(pic)
    arr = arr.flatten()

    print(arr)


if __name__ == "__main__":
    main()
