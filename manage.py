import streamlit as st
from PIL import Image as P_Image

import numpy as np
import cv2
import glob
import itertools
from IPython.display import display, Image
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from torchvision import transforms
import cshogi

# print('path =', os.getenv('PATH'))

# vipshome = 'c:\\vips-dev-8.7\\bin'
# os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

import torchmetrics
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from scipy.optimize import basinhopping

import base64

from model.model_structure import Net1, Net2

from yolov5 import detect

from tools.tool import *

# os.environ['path'] += r';C:\Users\skota\Development\projects\SHOGI_PHASE_RECOG_for_streamlit\GTK-for-Windows-Runtime-Environment-Installer\gtk-nsis-pack\bin'#デプロイ時はいらない？
import pyvips
# import svgwrite

# from cairosvg import svg2png
# import cairo
# import rsvg

#ネットワーク準備
net1 = Net1().cpu().eval()
net2 = Net2().cpu().eval()

# 重みの読み込み
net1.load_state_dict(torch.load('model/etl8g3.pt', map_location=torch.device('cpu')))
net2.load_state_dict(torch.load('model/koma1.pt', map_location=torch.device('cpu')))

# 推論モデルの駒ラベル番号とkoma_class_listのインデックス番号を一致させてリスト化。
koma_class_list = ["P", "S", "R", "B", "N", "G", "L", "+S", "+N", "+L", "K", "+R", "+P", "+B", "p", "s", "r", "b", "n", "g", "l", "+s", "+n", "+l", "k", "+r", "+p", "+b", "BLANK"]


st.title("将棋の局面認識アプリ")

left_column, right_column = st.columns(2)


#sidebar
# with st.sidebar.container():
#     st.sidebar.write("")
st.sidebar.subheader('局面画像をアップする　☟')
uploaded_file = st.sidebar.file_uploader("", type="jpg")
if uploaded_file is not None:
    image = P_Image.open(uploaded_file)
    image_array = np.array(image)
    left_column.subheader('入力画像: ')
    left_column.image(image_array, caption="up_image", use_column_width=True)
    img_name = "input.jpg"
    img_url = 'media/documents/{}'.format(img_name)
    image.save(img_url)

    # 入力画像をリモートに蓄積していく場合。
    # img_name = uploaded_file.name
    # img_url = 'media/documents/{}'.format(img_name)
    # image.save(img_url)
    
st.sidebar.caption("※　300ピクセル以上の画像を推奨")

with st.sidebar.container():
    st.sidebar.write("")
infer_button = st.sidebar.button("推論する")
latest_iteration = 0
bar = st.sidebar.progress(0)
bar.progress(latest_iteration)

with st.sidebar.container():
    st.sidebar.write("")
with st.sidebar.container():
    st.sidebar.write("")


#推論ボタン
if infer_button:
    #画像データをimgに格納

    latest_iteration = 2
    bar.progress(latest_iteration)

    files = [img_url]
    raw_imgs = [cv2.imread(f) for f in files]
    imgs = [fit_size(img, 500, 500) for img in raw_imgs]
    raw_img = raw_imgs[0]
    img = imgs[0]

    latest_iteration = 10
    bar.progress(latest_iteration)

    #盤の内側四角を選定
    rect, score = convex_poly_fitted(img)

    #4隅を元に盤の黒塗り画像を作成し保存（持ち駒特定用）
    polies = rect.astype('int32')
    points = np.array([(polies[0][0], polies[0][1]), (polies[1][0], polies[1][1]), (polies[2][0], polies[2][1]), (polies[3][0], polies[3][1])])
    img2 = cv2.fillConvexPoly(img, points, (0, 0, 0))
    bgr_img2 = img2[:, :, ::-1]
    im2 = P_Image.fromarray(bgr_img2)
    im2.save("media/documents/x_ban_black.jpg", quality=95)

    latest_iteration = 25
    bar.progress(latest_iteration)

    img_resized = img_resize(bgr_img2, "media/documents/x_ban_black.jpg")
    img_resized.save("media/documents/x_ban_black_resized.jpg")

    #結果画面に表示するinput画像（リサイズしたもの）を保存
    bgr_img3 = img[:, :, ::-1]
    img_resized = img_resize(bgr_img3, img_url)
    img_resized.save("media/documents/x_resized.jpg")

    #盤面部分をトリミング
    trimed = trim(raw_img, normalize_corners(rect) * (raw_img.shape[0] / img.shape[0]), False)

    #トリミング画像を保存
    lined = draw_ruled_line(trimed, False)
    cv2.imwrite('media/documents/x_ban_only.jpg', lined)

    #トリミング画像を各ピースに分解し格納
    cell_imgs = np.array(list(cells(trimed)))

    # 画像の前処理を定義
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    sfen = ""

    #各マスを推論しban_sfenに変換
    ban_sfen = ""
    BLANK_num = 0
    for i, cell in enumerate(cell_imgs):
        pil_img = P_Image.fromarray(cell)
        x = transform(pil_img)
        y = net2(x.unsqueeze(0))
        y = F.softmax(y)
        y = torch.argmax(y)
        y = y.item()
        koma_class = koma_class_list[y]

        a = (i + 1) % 9

        if (a == 0) & (i != 80):

            if koma_class == "BLANK":
                BLANK_num += 1
                ban_sfen += f"{BLANK_num}"
                ban_sfen += "/"
                BLANK_num = 0

            else:
                if BLANK_num == 0:
                    ban_sfen += koma_class
                    ban_sfen += "/"
                else:
                    ban_sfen += f"{BLANK_num}"
                    ban_sfen += koma_class
                    ban_sfen += "/"
                    BLANK_num = 0   


        else:

            if koma_class == "BLANK":
                BLANK_num += 1

            else:
                if BLANK_num == 0:
                    ban_sfen += koma_class
                else:
                    ban_sfen += f"{BLANK_num}"
                    ban_sfen += koma_class
                    BLANK_num = 0
    
    latest_iteration = 50
    bar.progress(latest_iteration)

    #持ち駒を検出、分類しmochi_sfenに変換
    mochi_sfen = ""
    if(os.path.isfile('yolov5/runs/detect/exp/labels/x_ban_black_resized.txt')):
        os.remove('yolov5/runs/detect/exp/labels/x_ban_black_resized.txt')

    latest_iteration = 60
    bar.progress(latest_iteration)

    opt = detect.parse_opt()
    detect.main(opt)

    latest_iteration = 80
    bar.progress(latest_iteration)

    path = "yolov5/runs/detect/exp/labels/x_ban_black_resized.txt"
    with open(path) as f:
        s = f.read()
    a = s.split()

    classified_list = []
    for classPlace in range(len(a) - 1):
        if classPlace % 5 == 0:
            classified_list.append(a[classPlace])

    mochi_sfen = classList_to_mochiSfen(classified_list)  

    sfen = ban_sfen + " b " + mochi_sfen
    board = cshogi.Board(sfen)
    x_resized_img_url = "media/documents/x_resized.jpg"

    svg_img = get_phase_svg(sfen)
    # print(sfen)
    # print(type(svg_img))
    # svg_img_bytes = bytes(svg_img, "utf-8")
    

    # svg2png(bytestring=svg_img, write_to='media/documents/x_result.png', output_width=500, output_height=400)

    # svg_pyvips_instance = pyvips.Image.svgload_buffer(svg_img_bytes, dpi=200)
    # svg_pyvips_instance.write_to_file('media/documents/x_result.png')

    html1 = get_html(svg_img)

    latest_iteration = 100
    bar.progress(latest_iteration)

    right_column.subheader('推論結果: ')
    # right_column.image('media/documents/x_result.png', caption="result", use_column_width=True)
    right_column.write(html1, unsafe_allow_html=True)

    st.session_state.key= html1


analysis_button = st.sidebar.button("推論途中の経過")
if analysis_button:
    if uploaded_file is None:
        st.sidebar.write("局面の推論が終わってから押してください")
    
    else:
        right_column.subheader('推論結果: ')#ここでも実行しないと、ボタン押したときに推論結果の画像がUIから消える
        # right_column.image('media/documents/x_result.png', caption="result", use_column_width=True)#ここでも実行しないと、ボタン押したときに推論結果の画像がUIから消える
        right_column.write(st.session_state.key, unsafe_allow_html=True)#ここでも実行しないと、ボタン押したときに推論結果の画像がUIから消える
        left_column2, right_column2 = st.columns(2)
        left_column2.subheader('盤抽出位置: ')
        right_column2.subheader('持ち駒検出（by YOLOv5）: ')

        ban_image = P_Image.open("media/documents/x_ban_only.jpg")
        ban_image_array = np.array(ban_image)
        left_column2.image(ban_image_array, caption="", use_column_width=True)

        yolo_image = P_Image.open("yolov5/runs/detect/exp/x_ban_black_resized.jpg")
        yolo_image_array = np.array(yolo_image)
        right_column2.image(yolo_image_array, caption="", use_column_width=True)

