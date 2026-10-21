from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "figures"
W, H = 1600, 900


def font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(38, True)
F_SUB = font(27, True)
F_TEXT = font(24)
F_SMALL = font(20)
F_TINY = font(17)


def canvas(title: str):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((55, 30), title, fill=(25, 35, 50), font=F_TITLE)
    return im, d


def arrow(d, a, b, fill, width=5, head=15):
    d.line([a, b], fill=fill, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for off in (math.pi * 0.82, -math.pi * 0.82):
        p = (b[0] + head * math.cos(ang + off), b[1] + head * math.sin(ang + off))
        d.line([b, p], fill=fill, width=width)


def rounded(d, box, fill, outline=None, width=2, radius=18):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fig1():
    im, d = canvas("PyBullet UR5 reaching task and pose representation")
    # ground and workspace
    d.rectangle((80, 670, 1515, 760), fill=(232, 236, 241), outline=(175, 182, 190), width=3)
    d.ellipse((280, 655, 600, 735), fill=(205, 211, 220), outline=(125, 135, 145), width=3)
    # stylized UR5 links
    pts = [(440, 650), (430, 520), (610, 425), (805, 485), (965, 355), (1120, 390)]
    links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for i, j in links:
        d.line([pts[i], pts[j]], fill=(70, 86, 105), width=42)
        d.line([pts[i], pts[j]], fill=(126, 145, 164), width=29)
    for x, y in pts:
        d.ellipse((x - 25, y - 25, x + 25, y + 25), fill=(45, 58, 75), outline=(220, 225, 230), width=3)
    d.rectangle((1090, 365, 1175, 415), fill=(55, 67, 83), outline=(25, 35, 45), width=3)
    d.text((295, 790), "fixed-base UR5", fill=(35, 45, 60), font=F_TEXT)
    # trajectory and current/goal poses
    traj = [(1128, 390), (1160, 370), (1192, 350), (1222, 328), (1250, 310), (1275, 290), (1300, 275)]
    d.line(traj, fill=(238, 153, 35), width=8)
    for p in traj[::2]:
        d.ellipse((p[0]-7, p[1]-7, p[0]+7, p[1]+7), fill=(238,153,35))
    d.ellipse((1272, 247, 1328, 303), fill=(42, 177, 104), outline=(20, 110, 65), width=3)
    d.text((1360, 300), "target", fill=(25, 115, 70), font=F_SMALL)
    # axes at target
    origin = (1300, 275)
    for end, col, lab in [((1400, 275), (210, 60, 60), "x_g"), ((1300, 175), (40, 145, 75), "y_g"), ((1345, 230), (45, 85, 200), "z_g")]:
        arrow(d, origin, end, col, width=5)
        d.text((end[0] + 8, end[1] - 13), lab, fill=col, font=F_SMALL)
    # three keypoints
    kp = [(1242, 250), (1260, 315), (1300, 300)]
    gkp = [(1285, 240), (1318, 295), (1350, 275)]
    for a, b in zip(kp, gkp):
        d.ellipse((a[0]-9, a[1]-9, a[0]+9, a[1]+9), fill=(224, 87, 56))
        d.ellipse((b[0]-9, b[1]-9, b[0]+9, b[1]+9), fill=(45, 120, 220))
        arrow(d, a, b, (130, 130, 130), width=3, head=10)
    d.text((1080, 470), "current keypoints", fill=(190, 65, 45), font=F_SMALL)
    d.text((1080, 505), "target keypoints", fill=(35, 95, 190), font=F_SMALL)
    # inset task definition
    rounded(d, (80, 110, 690, 315), (248, 250, 253), outline=(160, 170, 185), width=2)
    d.text((110, 140), "Task definition", fill=(25, 35, 50), font=F_SUB)
    lines = ["random position distance: 0.03–0.50 m", "random orientation difference: 0.03–π rad", "action: joint velocity command", "success: coupled pose error held for 5 steps"]
    for i, line in enumerate(lines):
        d.text((115, 190 + 30*i), "• " + line, fill=(55, 65, 80), font=F_TEXT)
    im.save(OUT / "fig1_ur5_task.png")


def fig2():
    im, d = canvas("Full method: local guidance, coupled objective, and replay coverage")
    boxes = [
        ((65, 220, 350, 420), "State", "pose errors\nkeypoint errors\nJacobian features", (225, 238, 250)),
        ((450, 220, 735, 420), "SAC actor", "joint-velocity\naction", (232, 245, 232)),
        ((835, 220, 1120, 420), "Precision scaling", "level-dependent\naction magnitude", (255, 242, 210)),
        ((1220, 220, 1510, 420), "PyBullet UR5", "pose transition\nand terminal outcome", (238, 230, 248)),
    ]
    for box, head, body, col in boxes:
        rounded(d, box, col, outline=(70, 85, 105), width=3)
        d.text((box[0]+22, box[1]+25), head, fill=(25, 35, 50), font=F_SUB)
        for i, line in enumerate(body.split("\n")):
            d.text((box[0]+22, box[1]+88+34*i), line, fill=(60, 70, 85), font=F_TEXT)
    for a, b in [((350, 320), (450, 320)), ((735, 320), (835, 320)), ((1120, 320), (1220, 320))]:
        arrow(d, a, b, (55, 70, 90), width=6)
    # enhancement modules
    modules = [
        ((120, 545, 515, 735), "Direction", "Keypoint Jacobian", (220, 238, 255)),
        ((605, 545, 995, 735), "Objective", "precision quality\nhold / leave\nbottleneck shaping", (255, 236, 210)),
        ((1085, 545, 1480, 735), "Data", "Success Replay\nBoundary Replay", (229, 245, 229)),
    ]
    for box, head, body, col in modules:
        rounded(d, box, col, outline=(80, 95, 110), width=3)
        d.text((box[0]+24, box[1]+22), head, fill=(25, 35, 50), font=F_SUB)
        for i, line in enumerate(body.split("\n")):
            d.text((box[0]+24, box[1]+78+34*i), line, fill=(60, 70, 85), font=F_TEXT)
    d.text((65, 800), "All methods share the same target distribution, SAC backbone, precision curriculum, and 6M-transition budget.", fill=(45, 55, 70), font=F_TEXT)
    im.save(OUT / "fig2_full_framework.png")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig3():
    im, d = canvas("Curriculum progression under a fixed training budget")
    left, top, right, bottom = 130, 145, 1510, 760
    d.rectangle((left, top, right, bottom), outline=(70, 80, 95), width=3)
    for x in range(0, 7):
        px = left + (right-left) * x / 6
        d.line((px, top, px, bottom), fill=(225, 229, 235), width=2)
        d.text((px-35, bottom+20), f"{x}M", fill=(55, 65, 80), font=F_SMALL)
    for lev in range(0, 14, 2):
        py = bottom - (bottom-top) * lev / 13
        d.line((left, py, right, py), fill=(235, 238, 242), width=2)
        d.text((left-48, py-12), f"L{lev}", fill=(55, 65, 80), font=F_SMALL)
    d.text((650, 842), "training transitions", fill=(45, 55, 70), font=F_TEXT)
    d.text((25, 285), "level", fill=(45, 55, 70), font=F_TEXT)
    configs = [
        ("Base", [ROOT/"outputs/retrain_20261013/kp_sac_weak/base_seed11001_t6/progress.csv", ROOT/"outputs/retrain_20261013/kp_sac_weak/base_seed11002_t6/progress.csv"], (100, 100, 100)),
        ("Full", [ROOT/"outputs/retrain_20261013/full/full_seed11001_6m_l13_resume/progress.csv", ROOT/"outputs/retrain_20261013/full/full_seed11002_6m_l13_resume/progress.csv"], (30, 105, 205)),
        ("M1", [ROOT/"outputs/retrain_20261013/m1_no_bottleneck/m1_seed11001_6m/progress.csv"], (220, 90, 45)),
        ("M2", [ROOT/"outputs/retrain_20261013/m2_no_boundary_replay/m2_seed11001_6m/progress.csv"], (40, 155, 85)),
        ("M3", [ROOT/"outputs/retrain_20261013/m3_no_jacobian/m3_seed11001_6m/progress.csv"], (150, 75, 165)),
    ]
    legend_y = 110
    for idx, (name, paths, col) in enumerate(configs):
        runs = []
        for path in paths:
            if not path.exists():
                continue
            rows = read_csv(path)
            pts = []
            for row in rows[::max(1, len(rows)//250)]:
                try:
                    step = float(row["global_step"]); level = float(row.get("orientation_level_index", row.get("goal_level_index", 0)))
                    pts.append((step, level))
                except (ValueError, KeyError):
                    pass
            if pts:
                runs.append(pts)
        if not runs:
            continue
        for pts in runs:
            pts.sort()
            xy = [(left + (right-left)*max(0,min(1,s/6_000_000)), bottom - (bottom-top)*max(0,min(13,l))/13) for s,l in pts]
            d.line(xy, fill=col, width=5, joint="curve")
        lx = 1100 + (idx%3)*135; ly = legend_y + (idx//3)*35
        d.line((lx, ly+12, lx+35, ly+12), fill=col, width=5)
        d.text((lx+45, ly), name, fill=(45, 55, 70), font=F_SMALL)
    d.text((170, 165), "higher precision level", fill=(75, 85, 100), font=F_SMALL)
    d.text((170, 720), "lower precision level", fill=(75, 85, 100), font=F_SMALL)
    im.save(OUT / "fig3_curriculum_progression.png")


def mean_heat(method, level):
    paths = sorted((ROOT/"outputs/evaluation_final").glob(f"{method}/L{level:02d}/eval_seed*.json"))
    arrays = []
    for path in paths:
        try:
            arrays.append(json.loads(path.read_text())["bin_success_rates"])
        except (KeyError, json.JSONDecodeError):
            pass
    if not arrays:
        return [0.0] * 100
    return [sum(a[i] for a in arrays) / len(arrays) for i in range(100)]


def heat(d, box, values, title):
    x0, y0, x1, y1 = box
    d.text((x0, y0-42), title, fill=(30, 40, 55), font=F_SUB)
    size = min((x1-x0), (y1-y0)) // 10
    for r in range(10):
        for c in range(10):
            v = max(0.0, min(1.0, values[r*10+c]))
            # white -> blue, with low success visually distinct
            col = (int(245-190*v), int(248-135*v), int(252-35*v))
            d.rectangle((x0+c*size, y0+r*size, x0+(c+1)*size-2, y0+(r+1)*size-2), fill=col)
            if size >= 45:
                d.text((x0+c*size+8, y0+r*size+8), f"{v:.1f}", fill=(25,35,50) if v<0.72 else "white", font=F_TINY)
    d.rectangle((x0, y0, x0+10*size, y0+10*size), outline=(75,85,100), width=2)


def fig4():
    im = Image.new("RGB", (W, 1550), "white")
    d = ImageDraw.Draw(im)
    d.text((55, 30), "Success-rate heatmaps over target distance and orientation bins", fill=(25, 35, 50), font=F_TITLE)
    heat(d, (120, 190, 720, 790), mean_heat("base_s11001", 6), "Base, L6")
    heat(d, (880, 190, 1480, 790), mean_heat("full_s11001", 6), "Full, L6")
    heat(d, (120, 850, 720, 1450), mean_heat("full_s11001", 13), "Full, L13")
    heat(d, (880, 850, 1480, 1450), mean_heat("m3_s11001", 11), "M3, L11")
    d.text((420, 1490), "target orientation bin (low → high)", fill=(45,55,70), font=F_SMALL)
    d.text((10, 455), "distance\nbin", fill=(45,55,70), font=F_SMALL)
    d.text((1180, 1490), "target orientation bin (low → high)", fill=(45,55,70), font=F_SMALL)
    d.text((770, 455), "distance\nbin", fill=(45,55,70), font=F_SMALL)
    im.save(OUT / "fig4_task_space_heatmaps.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4()
    print(f"wrote figures to {OUT}")
