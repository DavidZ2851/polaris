import numpy as np


def debug_plot(frame, pcd, K, T_cam_to_base):
    import cv2

    img = frame.copy()
    T_base_to_cam = np.linalg.inv(T_cam_to_base)

    def project(pts_world):
        pts_h = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)  # [N, 4]
        
        if T_base_to_cam.shape == (4, 4):
            pts_cam = (T_base_to_cam @ pts_h.T).T[:, :3]  # [N, 3]
        elif T_base_to_cam.shape == (3, 4):
            pts_cam = (T_base_to_cam @ pts_h.T).T  # [N, 3]
        else:
            # 3x3 rotation only, no translation
            pts_cam = (T_base_to_cam @ pts_world.T).T  # [N, 3]
        
        pts2d = (K @ pts_cam.T).T
        pts2d = (pts2d[:, :2] / pts2d[:, 2:3]).astype(int)
        return pts2d
    

    # ── Gripper PCD ───────────────────────────────────────────────────────────
    pts2d  = project(pcd)
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    labels = ["0", "1", "2", "3"]

    for pt, color, label in zip(pts2d, colors, labels):
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(img, tuple(pt), 5, color, -1)
            cv2.putText(img, label, tuple(pt + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if all(0 <= pts2d[i][0] < img.shape[1] and 0 <= pts2d[i][1] < img.shape[0] for i in [1, 2]):
        cv2.line(img, tuple(pts2d[1]), tuple(pts2d[2]), (255,255,255), 1)

    # ── World origin axes ─────────────────────────────────────────────────────
    axis_len = 0.1
    origins = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],  # X
        [0.0, axis_len, 0.0],  # Y
        [0.0, 0.0, axis_len],  # Z
    ])
    apts = project(origins)
    o = tuple(apts[0])
    ax_colors = [(0,0,255), (0,255,0), (255,0,0)]  # X=red, Y=green, Z=blue
    ax_labels = ["X", "Y", "Z"]
    for j, (ac, al) in enumerate(zip(ax_colors, ax_labels)):
        ep = tuple(apts[j+1])
        h, w = img.shape[:2]
        if all(0 <= p[0] < w and 0 <= p[1] < h for p in [o, ep]):
            cv2.arrowedLine(img, o, ep, ac, 2, tipLength=0.2)
            cv2.putText(img, al, ep, cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)

    return img