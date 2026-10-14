# Parse the provided custom-formatted text, create DataFrame, plot and save images
from pathlib import Path
import pandas as pd, io, matplotlib.pyplot as plt, os, re
csv_text = """epoch | train_loss | val_loss | mAP | mAP@0.3 | precision | recall | f1
EP001 | train_loss=0.1551 | val_loss=0.1282 | mAP=0.3698 | mAP@0.3=0.9857 | P=0.1333 | R=0.5000 | F1=0.2105
EP002 | train_loss=0.1080 | val_loss=0.0962 | mAP=0.4688 | mAP@0.3=1.0000 | P=0.2353 | R=0.5000 | F1=0.3200
EP003 | train_loss=0.0890 | val_loss=0.0857 | mAP=0.5539 | mAP@0.3=1.0000 | P=0.3636 | R=0.5000 | F1=0.4211
EP004 | train_loss=0.0831 | val_loss=0.0823 | mAP=0.5696 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP005 | train_loss=0.0808 | val_loss=0.0807 | mAP=0.5559 | mAP@0.3=1.0000 | P=1.0000 | R=1.0000 | F1=1.0000
EP006 | train_loss=0.0794 | val_loss=0.0798 | mAP=0.5749 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP007 | train_loss=0.0788 | val_loss=0.0794 | mAP=0.5887 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP008 | train_loss=0.0778 | val_loss=0.0781 | mAP=0.6134 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP009 | train_loss=0.0770 | val_loss=0.0772 | mAP=0.6339 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP010 | train_loss=0.0767 | val_loss=0.0773 | mAP=0.6236 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP011 | train_loss=0.0761 | val_loss=0.0766 | mAP=0.6741 | mAP@0.3=1.0000 | P=0.4444 | R=0.5000 | F1=0.4706
EP012 | train_loss=0.0754 | val_loss=0.0775 | mAP=0.6391 | mAP@0.3=1.0000 | P=1.0000 | R=1.0000 | F1=1.0000
EP013 | train_loss=0.0751 | val_loss=0.0752 | mAP=0.6636 | mAP@0.3=1.0000 | P=1.0000 | R=1.0000 | F1=1.0000
EP014 | train_loss=0.0742 | val_loss=0.0745 | mAP=0.6780 | mAP@0.3=1.0000 | P=1.0000 | R=1.0000 | F1=1.0000
"""

# parse lines
lines = [l.strip() for l in csv_text.strip().splitlines() if l.strip()]
rows = []
for line in lines[1:]:
    parts = [p.strip() for p in line.split("|")]
    ep_label = parts[0]
    ep_num = int(re.search(r"EP0*([0-9]+)", ep_label).group(1))
    # parse remaining key=value or P=,R=,F1=
    kv = {}
    for p in parts[1:]:
        # handle mAP@0.3 key name with @
        if "mAP@0.3" in p:
            k,v = p.split("=",1)
            k = "mAP03"
        else:
            if "=" in p:
                k,v = p.split("=",1)
                k = k.strip().lower()
            else:
                # handle P= etc
                k,v = p.split("=",1)
                k = k.strip().lower()
        kv[k.strip()] = v.strip()
    # build row with consistent column names
    row = {
        "epoch": ep_num,
        "train_loss": float(kv.get("train_loss", "nan")),
        "val_loss": float(kv.get("val_loss", "nan")),
        "map": float(kv.get("map", "nan")),
        "mAP03": float(kv.get("mAP03", kv.get("mAP@0.3", "nan"))),
        "prec": float(kv.get("p", kv.get("precision", "nan"))),
        "rec": float(kv.get("r", kv.get("recall", "nan"))),
        "f1": float(kv.get("f1", "nan"))
    }
    rows.append(row)

df = pd.DataFrame(rows).sort_values('epoch').reset_index(drop=True)

outdir = Path(r"D:\PMT_Paper_Fasterrcnn-resnet50\output\plots")
outdir.mkdir(parents=True, exist_ok=True)
csv_out = outdir / "metrics_history_custom.csv"
df.to_csv(csv_out, index=False)

# Plotting following rules
epochs = df['epoch'].tolist()

# Loss curve
plt.figure(figsize=(9,5))
plt.plot(epochs, df['train_loss'], marker='o', label='train_loss')
plt.plot(epochs, df['val_loss'], marker='o', label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss curve'); plt.grid(True); plt.legend()
plt.tight_layout()
loss_path = outdir / "loss_curve_custom.png"
plt.savefig(loss_path, dpi=160); plt.close()

# mAP curves
plt.figure(figsize=(9,5))
plt.plot(epochs, df['map'], marker='o', label='mAP@0.5:0.95')
plt.plot(epochs, df['mAP03'], marker='o', label='mAP@0.3')
plt.xlabel('Epoch'); plt.ylabel('mAP'); plt.title('mAP curves'); plt.grid(True); plt.legend()
plt.tight_layout()
map_path = outdir / "map_curve_custom.png"
plt.savefig(map_path, dpi=160); plt.close()

# PRF curve
plt.figure(figsize=(9,5))
plt.plot(epochs, df['prec'], marker='o', label='precision')
plt.plot(epochs, df['rec'], marker='o', label='recall')
plt.plot(epochs, df['f1'], marker='o', label='f1')
plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Precision / Recall / F1 (pos)'); plt.grid(True); plt.legend()
plt.tight_layout()
prf_path = outdir / "prf_curve_custom.png"
plt.savefig(prf_path, dpi=160); plt.close()

# Save small zoom of first 8 epochs loss
plt.figure(figsize=(9,5))
sub = df[df['epoch'] <= 8]
plt.plot(sub['epoch'], sub['train_loss'], marker='o', label='train_loss')
plt.plot(sub['epoch'], sub['val_loss'], marker='o', label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss curve (Epoch 1-8)'); plt.grid(True); plt.legend()
plt.tight_layout()
loss_zoom_path = outdir / "loss_curve_1_8_custom.png"
plt.savefig(loss_zoom_path, dpi=160); plt.close()

saved_files = sorted([str(p) for p in outdir.glob("*")])
print("Saved files:")
print("\n".join(saved_files))

# display table to user
import caas_jupyter_tools as tools
tools.display_dataframe_to_user("ssd_metrics_custom", df)

# Provide file links
for p in saved_files:
    print("file://" + p)

