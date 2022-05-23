import cv2
import numpy as np
import itertools
from scipy.optimize import basinhopping
from PIL import Image as P_Image
import math
import cshogi

def fit_size(img, h, w):
    size = img.shape[:2]
    f = min(h / size[0], w / size[1])
    return cv2.resize(img, (int(size[1] * f), int(size[0] * f)), interpolation=cv2.INTER_AREA)


# 輪郭検出準備
def edge(img, show=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# BGR to GRAY (cv2.imreadで読み込んだファイルはBGRになってしまう。後の直線検出のためグレー化)
    edges = cv2.Canny(gray, 50, 150)#エッジ検出。第2引数は目的エッジの他エッジとの隣接部分がエッジかの閾値。第3引数は目的エッジがエッジかの閾値。
    if show:
        display_cv_image(edges)
    return edges

def line(img, show=True, threshold=80, minLineLength=50, maxLineGap=5):
    edges = edge(img, False)#場合によってはHoughLinesPの処理の前（この行）で画像を白黒反転させる必要あり。
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 200, minLineLength, maxLineGap)#直線検出。線の座標がndarrayで返される。
    #第2引数は直角座標点と直線の距離（あまり調整する必要なし）
    #第3引数は直角座標点と直線の角度 (あまり調整する必要なし)
    #第4引数は直線とみなすための閾値（何ピクセルで直線とみなすか？）
    #第5引数は直線とみなす最小の長さ
    #第6引数は同一直線とみなす点間隔の長さ
    if show:
        blank = np.zeros(img.shape, np.uint8)
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 1)
            #第2引数は直線の始点
            #第3引数は直線の終点
            #第4引数はカラー
            #第5引数は線の太さ
        display_cv_image(blank)
    return lines

def contours(img, show=True):#輪郭
    edges = edge(img, False)
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]#画像から輪郭を抽出する
    #第2引数はmode。cv2.PETR_LISTは検出したすべての輪郭に階層構造を与えない
    #第3引数はmethod。CV_CHAIN_APPROX_SIMPLEは水平・垂直・斜めの成分を圧縮して端点として格納。
    blank = np.zeros(img.shape, np.uint8)
    min_area = img.shape[0] * img.shape[1] * 0.2 # 画像の何割占めるか
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]#cv2.contourArea()は輪郭の面積を計算
    cv2.drawContours(blank, large_contours, -1, (0,255,0), 1)#輪郭を描く
    #第3引数は輪郭を形成する画素(点)のインデックス番号を指定する。
    #例えば0を指定すると、1番目の輪郭を形成する画素(点)のみを描画する。1を指定すると、2番目の輪郭を形成する画素(点)のみを描画する。
    #輪郭を形成する画素(点)を全て描画したい場合は、-1を指定する。
    if show:
        display_cv_image(blank)
    return large_contours

def convex(img, show=True):#凸
    blank = np.copy(img)
    convexes = []
    for cnt in contours(img, False):
        convex = cv2.convexHull(cnt)#凸包を実施。全ての点を含む最小多角形を求める。
        cv2.drawContours(blank, [convex], -1, (0,255,0), 2)
        convexes.append(convex)
    if show:
        display_cv_image(blank)
    return convexes

def convex_poly(img, show=True):
    cnts = convex(img, False)
    blank = np.copy(img)
    polies = []
    for cnt in cnts:
        arclen = cv2.arcLength(cnt, True)#輪郭の周囲の長さ
        poly = cv2.approxPolyDP(cnt, 0.005*arclen, True)#輪郭の近似
        #第2引数はepsilonと呼ばれ、実際の輪郭と近似輪郭の最大距離
        cv2.drawContours(blank, [poly], -1, (0,255,0), 2)
        polies.append(poly)
    if show:
        display_cv_image(blank)
    return [poly[:, 0, :] for poly in polies]

def show_fitted(img, x):
    cntr = np.int32(x.reshape((4, 2)))
    blank = np.copy(img)
    cv2.drawContours(blank, [cntr], -1, (0,255,0), 2)
    display_cv_image(blank)
    
def select_corners(img, polies):
    p_selected = []
    p_scores = []
    for poly in polies:
        choices = np.array(list(itertools.combinations(poly, 4)))#要素の内、長さ4の部分列を返す
        scores = []
        # 正方形に近いものを選ぶ
        for c in choices:
            line_lens = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
            base = cv2.contourArea(c) ** 0.5
            score = sum([abs(1 - l/base) ** 2 for l in line_lens])
            scores.append(score)
        idx = np.argmin(scores)
        p_selected.append(choices[idx])
        p_scores.append(scores[idx])
    return p_selected[np.argmin(p_scores)]


# 輪郭検出後、内側の矩形を選択を準備
def gen_score_mat():
    half_a = np.fromfunction(lambda i, j: ((10 - i) ** 2) / 100.0, (10, 20), dtype=np.float32)
    half_b = np.rot90(half_a, 2)
    cell_a = np.r_[half_a, half_b]
    cell_b = np.rot90(cell_a)
    cell = np.maximum(cell_a, cell_b)
    return np.tile(cell, (9, 9))

SCALE = 0.7
def get_get_fit_score(img, x):
    # 入力リサイズ
    img = cv2.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
    img_size = (img.shape[0] * img.shape[1]) ** 0.5
    x = np.int32(x * SCALE)

    # 線分化
    poly_length = cv2.arcLength(x, True)
    lines = line(img, False, int(poly_length / 12), int(poly_length / 200))
    line_mat = np.zeros(img.shape, np.uint8)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_mat, (x1, y1), (x2, y2), 255, 1)
    line_mat = line_mat[:, :, 0]
    
    # 矩形の外をマスクアウト
    img_size = (img.shape[0] * img.shape[1]) ** 0.5
    mask = np.zeros(line_mat.shape, np.uint8)
    cv2.fillConvexPoly(mask, x, 1)
    kernel = np.ones((int(img_size / 10), int(img_size / 10)), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    line_mat[np.where(mask == 0)] = 0
    
    # スコア
    score_mat = gen_score_mat()

    def get_fit_score(x):
        img_pnts = np.float32(x).reshape(4, 2)
        img_pnts *= SCALE
        score_size = score_mat.shape[0]
        score_pnts = np.float32([[0, 0], [0, score_size], [score_size, score_size], [score_size, 0]])

        transform = cv2.getPerspectiveTransform(score_pnts, img_pnts)
        score_t = cv2.warpPerspective(score_mat, transform, (img.shape[1], img.shape[0]))

        res = line_mat * score_t
        return -np.average(res[np.where(res > 255 * 0.1)])
    
    return get_fit_score

def convex_poly_fitted(img, show=False):
    polies = convex_poly(img, False)
    poly = select_corners(img, polies)
    x0 = poly.flatten()
    get_fit_score = get_get_fit_score(img, poly)
    ret = basinhopping(get_fit_score, x0, T=0.1, niter=300, stepsize=3)
    if show:
        show_fitted(img, ret.x)
    return ret.x.reshape(4, 2), ret.fun


#盤面トリミングを準備
def normalize_corners(v):
    rads = []
    for i in range(4):
        a = v[(i + 1) % 4] - v[i]
        a = a / np.linalg.norm(a)
        cosv = np.dot(a, np.array([1, 0]))
        rads.append(math.acos(cosv))
    left_top = np.argmin(rads)
    return np.roll(v, 4 - left_top, axis=0)

base_size = 32
def trim(img, corners, show=True):
    w = base_size * 14
    h = base_size * 15
    transform = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    normed = cv2.warpPerspective(img, transform, (w, h))
    if show:
        display_cv_image(normed)
    return normed

def draw_ruled_line(img, show=True):
    w = base_size * 14
    h = base_size * 15
    img = img.copy()
    for i in range(10):
        x = int((w / 9) * i)
        y = int((h / 9) * i)
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    if show:
        display_cv_image(img)
    return img


#トリミング画像を各ピースに分解し格納 を準備
img_rows, img_cols = 48, 48
def cells(img):
    dx = img.shape[0] / 9
    dy = img.shape[1] / 9
    for i in range(9):
        for j in range(9):
            sx = int(dx * i)
            sy = int(dy * j)
            yield normalize(img[sx:(int(sx + dx)), sy:(int(sy + dy))], img_rows, img_cols)

def normalize(img, h, w):
    size = img.shape[:2]
    f = min(h / size[0], w / size[1])
    resized = cv2.resize(img, (int(size[1] * f), int(size[0] * f)), interpolation=cv2.INTER_AREA)

    color = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return color


def img_resize(img, path_part):
  img3 = P_Image.open(path_part)
  height = img.shape[0]
  width = img.shape[1]
  if width >= height:
    a = 640
    b = int(height * 640/width)
    return img3.resize((a, b))
  if width < height:
    a = int(width * 640/height)
    b = 640
    return img3.resize((a, b))


def get_phase_svg(sfen):
    board = cshogi.Board(sfen)
    svg = board.to_svg()
    sfen = ""#2回目のアップロード時に1回目のsfen末尾に連結されないように初期化
    # return HttpResponse(svg, content_type='image/svg+xml')
    return svg


def class_to_mochiSfenPiece(num, class_str):
  if num == 0:
    mochi_sfenPiece = ""
    return mochi_sfenPiece
  elif num == 1:
    mochi_sfenPiece = (class_str)
    return mochi_sfenPiece
  else:
    mochi_sfenPiece = (f"{num}"  + class_str)
    return mochi_sfenPiece


def classList_to_mochiSfen(c_list):
  mochi_sfen = ""
  # YOLO学習時に自分で設定したclass順に取得
  P_num = c_list.count('0')
  S_num = c_list.count('1')
  R_num = c_list.count('2')
  B_num = c_list.count('3')
  N_num = c_list.count('4')
  G_num = c_list.count('5')
  L_num = c_list.count('6')
  p_num = c_list.count('7')
  s_num = c_list.count('8')
  r_num = c_list.count('9')
  b_num = c_list.count('10')
  n_num = c_list.count('11')
  g_num = c_list.count('12')
  l_num = c_list.count('13')

  # 持ち駒のsfenにも記載順がある 飛車⇒角⇒...
  mochi_sfen += class_to_mochiSfenPiece(R_num, "R")
  mochi_sfen += class_to_mochiSfenPiece(B_num, "B")
  mochi_sfen += class_to_mochiSfenPiece(G_num, "G")
  mochi_sfen += class_to_mochiSfenPiece(S_num, "S")
  mochi_sfen += class_to_mochiSfenPiece(N_num, "N")
  mochi_sfen += class_to_mochiSfenPiece(L_num, "L")
  mochi_sfen += class_to_mochiSfenPiece(P_num, "P")
  mochi_sfen += class_to_mochiSfenPiece(r_num, "r")
  mochi_sfen += class_to_mochiSfenPiece(b_num, "b")
  mochi_sfen += class_to_mochiSfenPiece(g_num, "g")
  mochi_sfen += class_to_mochiSfenPiece(s_num, "s")
  mochi_sfen += class_to_mochiSfenPiece(n_num, "n")
  mochi_sfen += class_to_mochiSfenPiece(l_num, "l")
  mochi_sfen += class_to_mochiSfenPiece(p_num, "p")
  return mochi_sfen




