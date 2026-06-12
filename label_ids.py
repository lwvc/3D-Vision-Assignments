import colorsys
import cv2
import numpy as np
import os
import argparse

def build_lut(max_id=256):
    golden_ratio = 1.6180339887
    lut = {}
    for i in range(1, max_id):
        h = ((i * golden_ratio) % 1)
        s = 0.5 + (i % 2) * 0.5
        r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
        lut[(int(r*255), int(g*255), int(b*255))] = i
    return lut

def label_image(pred_path, render_path, lut):
    img = cv2.imread(pred_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for color in np.unique(img_rgb.reshape(-1, 3), axis=0):
        rgb = tuple(color)
        if sum(rgb) < 10:
            continue
        obj_id = lut.get(rgb)
        if obj_id is None:
            continue

        bgr = (rgb[2], rgb[1], rgb[0])
        mask = cv2.inRange(img, np.array(bgr), np.array(bgr))
        if cv2.countNonZero(mask) == 0:
            continue

        # 找所有連通區域，每個區域單獨標 ID
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < 200:  # 太小的色塊跳過
                continue

            # 在這個連通區域內找最大內切點（比中心點更準）
            comp_mask = (labels == comp_id).astype(np.uint8)
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = max_loc
            # 確保文字不超出畫布
            cx = max(15, min(cx, img.shape[1] - 25))
            cy = max(10, min(cy, img.shape[0] - 5))

            text = str(obj_id)
            text_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            cv2.putText(img, text, (cx-10, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)       # 黑色邊框
            cv2.putText(img, text, (cx-10, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)    # 色塊顏色字

    render = cv2.imread(render_path)
    h, w = img.shape[:2]
    render = cv2.resize(render, (w, h))
    return np.hstack([render, img])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="ours_XXXXX 資料夾路徑")
    args = parser.parse_args()

    pred_dir = os.path.join(args.input, "objects_pred")
    render_dir = os.path.join(args.input, "renders")
    out_dir = os.path.join(args.input, "id_labeled")
    os.makedirs(out_dir, exist_ok=True)

    lut = build_lut()
    files = sorted(os.listdir(pred_dir))

    for fname in files:
        pred_path = os.path.join(pred_dir, fname)
        render_path = os.path.join(render_dir, fname)
        if not os.path.exists(render_path):
            continue
        result = label_image(pred_path, render_path, lut)
        cv2.imwrite(os.path.join(out_dir, fname), result)
        print(f"處理: {fname}")

    print(f"\n完成！共 {len(files)} 張，存到 {out_dir}")
