from __future__ import division
from pylab import *
import numpy as np
import cv2


def thresh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def closing(image):
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=7)
    return closing


def get_size(cnt, which):
    mx = []
    for i in cnt:
        for x in i:
            mx.append(x[which])
    return np.max(mx) - np.min(mx)


def change_sv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s2 = np.ones_like(s) * 255
    v2 = np.ones_like(v) * 255
    h11 = cv2.merge((h, s2, v2))
    color = cv2.cvtColor(h11, cv2.COLOR_HSV2BGR)
    return color


def median_color(img):
    img = change_sv(img)
    b, g, r = cv2.split(img)
    return np.median(b), np.median(g), np.median(r)


def mean_color(img):
    b, g, r = cv2.split(img)
    return np.mean(b), np.mean(g), np.mean(r)


def without(tab):
    tabs = []
    for x in range(len(tab)):
        for y in range(len(tab[x])):
            if tab[x, y]<254 and tab[x, y]>1:
                tabs.append(tab[x, y].copy())
    return np.mean(tabs)


def median_cut_out(img): # mediana koloru, nie liczac czarnego
    piksele = []
    for x in range(len(img)):
        for y in range(len(img[x])):
            b, g, r = img[x][y]
            if (b+g+r) != 0:
                piksele.append([b, g, r])
    median = np.median(piksele)
    return median


def change_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def change_contrast(img):
    temp = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img)

    MIN = 75
    y_eq = y - MIN
    y_eq[y_eq > 255 - MIN] = 0
    y_eq = change_gamma(y_eq, 1.25)
    ycrcb = cv2.merge((y_eq, cr, cb))
    contr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return contr


def count_edge(cnt):
    arc_len = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.1 * arc_len, True)
    return len(approx)


def jedynka(imgo, nr):
    imgo = cv2.medianBlur(imgo, 3)
    wys, szer = imgo.shape
    sy, sx = wys/2, szer/2
    wynik, kat_obrotu = 0, 0
    srednia_wynikow = 0
    for i in range(0,36,1):
        M = cv2.getRotationMatrix2D((sx, sy), i*10, 1)
        img = cv2.warpAffine(imgo, M, (szer, wys))
        a = sx - (szer * 0.275)
        b = sx + (szer * 0.3)
        if wys < 300:
            c = sy + (wys * 0.14)
        elif wys < 500:
            c = sy + (wys * 0.13)
        else:
            c = sy + (wys * 0.11)
        d = sy + (wys * 0.27)

        zloty = img[int(c):int(d), int(a):int(b)].copy()
        ile_zl = (d - c) * (b - a)

        a = sx - (szer * 0.125)
        b = sx + (szer * 0.125)
        c = sy - (wys * 0.35)
        if wys < 300:
            d = sy + (wys * 0.11)
        else:
            d = sy + (wys * 0.1)
        jedynka = img[int(c):int(d), int(a):int(b)].copy()
        ile_jed = (d - c) * (b - a)

        reszta = img.copy()

        rsuma = 0
        rile = 0
        max_odl = int(0.7 * min(wys/2, szer/2))
        for index, val in np.ndenumerate(reszta):
            x2, y2 = index
            odl = math.hypot(x2 - sx, y2 - sy)
            if odl < max_odl:
                rsuma += val
                rile += 1

        sum_zl = cv2.sumElems(zloty)
        sum_jed = cv2.sumElems(jedynka)
        ssrednia = (sum_zl[0] + sum_jed[0]) / (ile_zl + ile_jed)
        rsrednia = (rsuma - sum_zl[0] - sum_jed[0]) / (rile - ile_zl - ile_jed)
        srednia_wynikow += (ssrednia - rsrednia)
        if ssrednia - rsrednia > wynik:
            wynik = ssrednia - rsrednia
            kat_obrotu = i*10
    srednia_wynikow /= 36
    M = cv2.getRotationMatrix2D((sx, sy), kat_obrotu, 1)
    img_ost = cv2.warpAffine(imgo, M, (szer, wys))
    jeden = img_ost[int(c):int(d), int(a):int(b)].copy()
    return abs(srednia_wynikow - wynik),M,szer,wys


def make_contours(image, closing, thresh, temp):
    _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moneta = 0
    monety = []
    liczba_banknotow = 0
    suma = 0
    colors = []
    number=0
    for cnt in contours:
        width = get_size(cnt, 0)
        height = get_size(cnt, 1)

        if width < 150:
            continue

        (_, _), (w, h), _ = cv2.fitEllipse(cnt)
        number += 1

        if abs(w - h) < 35:
            print("\nMoneta nr: " + str(moneta + 1))
            moneta += 1
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (w, h), c = ellipse

            cut_in = image[int(y - h / 3):int(y + h / 3), int(x - w / 3):int(x + w / 3)].copy()
            cut_in2 = image[int(y - h / 3):int(y + h / 3), int(x - w / 3):int(x + w / 3)].copy()
            srodek = image[int(y)-20:int(y)+20,int(x)-20:int(x)+20].copy()
            srodek = cv2.blur(srodek, (5, 5))

            rog = cut_in[0:40,0:40].copy()
            rog = cv2.blur(rog, (5, 5))
            rog = cv2.blur(rog, (5, 5))

            yccs = cv2.cvtColor(srodek, cv2.COLOR_BGR2YCrCb)
            ys, tcrs, tcbs = cv2.split(yccs)
            yccr = cv2.cvtColor(rog, cv2.COLOR_BGR2YCrCb)
            yr, tcrr, tcbr = cv2.split(yccr)

            crs, crr = mean(tcrs), mean(tcrr)
            cbs, cbr = mean(tcbs), mean(tcbr)

            #print(crs, crr)
            #print(cbs, cbr)


            if abs(crs - crr) > 3.5 and abs(cbs - cbr) > 8: # DWUKOLOROWE
                if crs < crr and cbs > cbr:
                    print("2zl")
                    suma += 2.0
                    image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (0, 255, 0)
                elif crs > crr and cbs < cbr:
                    print("5zl")
                    suma += 5.0
                    image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = 0
            elif abs(crs - crr) < 3.5 and abs(cbs - cbr) < 8: # JEDNOKOLOROWE
                if ((crs+crr)/2) > 139 and ((cbs+cbr)/2) < 111: # ZLOTE
                    print("5gr")
                    suma += 0.05
                    image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (0, 0, 255)
                else: # SREBRNE

                    wys = int(y + h / 3) - int(y - h / 3)
                    sze = int(x + w / 3) - int(x - w / 3)
                    wys = 0.7 * wys / 2
                    sze = 0.7 * sze / 2
                    max_odl = int(min(wys, sze))
                    srodek_con = thresh[int(y - h / 3):int(y + h / 3), int(x - w / 3):int(x + w / 3)].copy()
                    x1, y1= srodek_con.shape
                    x1, y1 = x1 / 2, y1 / 2
                    for index, val in np.ndenumerate(srodek_con):
                        x2, y2 = index
                        odl = math.hypot(x2 - x1, y2 - y1)
                        if odl > max_odl:
                            srodek_con[x2][y2] = 0
                    pkt_jedynki, M, szer, wys = jedynka(srodek_con, moneta)
                    #print(pkt_jedynki)

                    if int(pkt_jedynki) > 31:
                        print("1zl")
                        suma += 1.0
                        image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (255, 0, 255)
                    elif int(pkt_jedynki) < 15:
                        print("20gr")
                        suma += 0.2
                        image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (255, 255, 0)
                    else:
                        cv2.imwrite('./wyniki/moneta' + str(moneta) + '.jpg', cut_in2)
                        temp2 = cv2.imread('./wyniki/moneta' + str(moneta) + '.jpg', 0)
                        result = []
                        for tmp in temp:
                            tmp2 = cv2.resize(temp2.copy(), (len(tmp[0]), len(tmp)))
                            srednia1 = np.average(tmp)
                            tmp = (tmp > srednia1) * 255
                            srednia2 = np.average(tmp2)
                            tmp2 = (tmp2 > srednia2) * 255
                            res = (cv2.countNonZero(cv2.absdiff(tmp, tmp2)))
                            wynik = 1 - int(res) / (len(tmp[0]) * len(tmp))
                            result.append(wynik)
                        #print(result)
                        wynik = result.index(np.max(result))
                        if wynik < 2:
                            print("10gr")
                            suma += 0.1
                            image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (255, 0, 0)
                        elif wynik < 5:
                            print("20gr")
                            suma += 0.2
                            image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (255, 255, 0)
                        elif wynik < 7:
                            print("1zl")
                            suma += 1.0
                            image[int(y) - 15:int(y) + 15, int(x) - 15:int(x) + 15] = (255, 0, 255)


            cv2.imwrite('./wyniki/moneta' + str(moneta) + '.jpg', cut_in2)

            if len(cut_in) > 0:
                b, g, r = median_color(cut_in)
            colors.append([np.median(b),np.median(g),np.median(r)])
            cut_out = image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)].copy()
            width_cut, height_cut, _ = cut_out.shape
            rad_w, rad_h = int(width_cut/2), int(height_cut/2)
            cv2.circle(cut_out, (rad_w, rad_h), max(rad_w, rad_h), (0, 0, 0), -1)
            med_in, med_out = np.median(cut_in), median_cut_out(cut_out)
            monety.append(width * height)
            cv2.ellipse(image, ellipse, (255, 0, 0), 20)
        else:
            if width*height < 500000:
                continue
            points = 0
            box = get_box(cnt)
            cut = four_point_transform(image, box).copy()
            cut_k = four_point_transform(closing, box).copy()
            cut_k = cut_k[50:-50, 50:-50]
            _, cont2, _ = cv2.findContours(cut_k, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print("\nBanknot nr: " + str(liczba_banknotow+1))

            for cnt2 in cont2:
                width = get_size(cnt2, 0)
                height = get_size(cnt2, 1)
                if width < 100 or abs(width-height) > 20:
                    continue
                # sprawdzenie czy jest kółko na banknocie
                if is_circle(cnt2,cut_k):
                    points -= 1
                else:
                    points += 1

                # sprawdzenie czy jest kwadrat na banknocie
                if is_square(cnt2, cut_k):
                    points += 1
                else:
                    points -= 1

            # sprawdzenie mediany koloru banknotu
            if len(cut) > 0:
                b, g, r = median_color(cut)
                if g > 140:
                    points += 1
                else:
                    points -= 1

            decyzja = 0
            if points > 0:
                suma += 10
                decyzja = 10
            elif points < 0:
                suma += 20
                decyzja = 20
            else:
                continue
            liczba_banknotow += 1
            if decyzja == 10:
                cv2.drawContours(image, [box], 0, (0, 0, 255), 20)
            elif decyzja == 20:
                cv2.drawContours(image, [box], 0, (0, 255, 0), 20)
            cv2.imwrite('./wyniki/banknot' + str(liczba_banknotow) + '.jpg', cut)

    text = "Liczba monet: " + str(moneta)
    cv2.putText(image, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10)
    text = "Liczba banknotow: " + str(liczba_banknotow)
    cv2.putText(image, text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10)
    text = "Suma: " + "{:.2f}".format(suma) + " zlotych"
    cv2.putText(image, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 20)
    return image


def dlugosc(tab):
    return np.max(tab)-np.min(tab)


def srebrniki(img):
    ilosc = 0
    for x in range(len(img)):
        for y in range(len(img[x])):
            if img[x, y] > 128:
                ilosc += 1
    return ilosc/(len(img)*len(img[0]))


def get_box(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def is_square(cnt, img):
    y = []
    x = []
    arc_len = cv2.arcLength(cnt, True)
    approx2 = cv2.approxPolyDP(cnt, 0.1 * arc_len, True)

    for app in approx2:
        for app2 in app:
            x.append(app2[1])
            y.append(app2[0])
    x1, x2, y1, y2 = np.min(x) + 12, np.max(x) - 12, np.min(y) + 12, np.max(y) - 12
    count = 0
    for x in [x1, x2]:
        for y in [y1, y2]:
            if img[x, y] == 255:
                count += 1
    if count>2:
        return True
    else:
        return False


def is_circle(cnt, img):
    box = get_box(cnt)
    X = four_point_transform(img, box).copy()
    circles = cv2.HoughCircles(X, cv2.HOUGH_GRADIENT, 1.5, 5)
    if not circles is None:
        return True
    else:
        return False


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    org_img = cv2.imread("./wejscie/bm_all-x.jpg")
    temp=[]
    temp.append(cv2.imread('./zdjecia/moneta14.jpg', 0))
    temp.append(cv2.imread('./zdjecia/moneta15.jpg', 0))
    temp.append(cv2.imread('./zdjecia/moneta17.jpg', 0))
    temp.append(cv2.imread('./zdjecia/moneta19.jpg', 0))
    temp.append(cv2.imread('./zdjecia/moneta4.jpg', 0))
    temp.append(cv2.imread('./zdjecia/moneta20.jpg', 0))
    temp.append(cv2.imread('./zdjecia/moneta211.jpg', 0))
    kontrast = change_contrast(org_img)
    img = org_img
    thresh = thresh(org_img)
    closing = closing(thresh)
    img = make_contours(org_img, closing,thresh, temp)

    #cv2.imwrite('./wyniki/kontrast.jpg',kontrast)
    #cv2.imwrite("./wyniki/thresh.jpg", thresh)
    #cv2.imwrite("./wyniki/close.jpg", closing)
    cv2.imwrite('./wyniki/wynik.jpg', img)