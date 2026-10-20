#!/bin/bash
# Builds "FL DoS Swarm v14.app": a double-clickable launcher that opens the dashboard in its own desktop
# window (desktop_app.py, pywebview) from this project's .venv. Re-run after moving the project folder.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")" && pwd)"
APP="$PROJECT/FL DoS Swarm v14.app"
PORT=8502

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FL DoS Swarm v14</string>
  <key>CFBundleDisplayName</key><string>FL DoS Swarm v14</string>
  <key>CFBundleIdentifier</key><string>lk.sliit.fldosswarm.v14</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict></plist>
PLIST

cat > "$APP/Contents/MacOS/launcher" <<LAUNCH
#!/bin/bash
# opens the dashboard in its own desktop window (desktop_app.py starts/stops the server on port $PORT)
cd "$PROJECT" || exit 1
export FLDOS_PORT=$PORT
exec "$PROJECT/.venv/bin/python" "$PROJECT/desktop_app.py"
LAUNCH
chmod +x "$APP/Contents/MacOS/launcher"

# Icon: drawn with matplotlib, converted with iconutil.
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
"$PROJECT/.venv/bin/python" - "$ICONSET" <<'PY'
import sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
out = sys.argv[1]
fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.08, 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.18", fc="#16213e"))
centre = (0.5, 0.5)
pts = [(0.5 + 0.27 * math.cos(a), 0.5 + 0.27 * math.sin(a)) for a in [math.pi / 2 + k * 2 * math.pi / 6 for k in range(6)]]
for i, p in enumerate(pts):
    ax.plot([centre[0], p[0]], [centre[1], p[1]], color="#4f8cff", lw=6, alpha=0.8, zorder=1)
    colour = "#2ecc71" if i == 1 else "#e74c3c" if i == 4 else "#cfd8ff"
    ax.add_patch(Circle(p, 0.06, fc=colour, zorder=2))
ax.add_patch(Circle(centre, 0.1, fc="#4f8cff", zorder=3))
ax.text(0.5, 0.5, "FL", ha="center", va="center", color="white", fontsize=70, weight="bold", zorder=4)
fig.savefig(f"{out}/icon_512x512@2x.png", transparent=True)
from PIL import Image
im = Image.open(f"{out}/icon_512x512@2x.png")
for s in (16, 32, 128, 256, 512):
    im.resize((s, s), Image.LANCZOS).save(f"{out}/icon_{s}x{s}.png")
    im.resize((2 * s, 2 * s), Image.LANCZOS).save(f"{out}/icon_{s}x{s}@2x.png")
PY
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
touch "$APP"
echo "Built: $APP"
echo "Drag it to /Applications or the Dock if you like (it still runs code from $PROJECT)."
