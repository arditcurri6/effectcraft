#!/usr/bin/env bash
# Build a local development app, EffectCraft-Custom.app, next to (never over) an installed
# EffectCraft.app:
#
#   - its own bundle identifier (ai.storyteller.effectcraft.custom), name and executable, so macOS
#     treats it as a separate application;
#   - its own data folder (~/Library/Application Support/EffectCraft Custom: settings,
#     workspaces, crash recovery, CineLooks user looks), set at compile time with
#     EFFECTCRAFT_DATA_NAME;
#   - it opens .ecproj files as an alternate handler and imports (does not export) the project
#     type, so the installed release stays the owner;
#   - ad-hoc signed with the release entitlements. Not notarized: for this Mac only.
#
# Usage: packaging/macos/custom-app.sh [--dest DIR] [--skip-build]
#   --dest DIR     where to put the app (default ~/Applications); an existing
#                  EffectCraft-Custom.app there is replaced, nothing else is touched.
#   --skip-build   reuse target/aarch64-apple-darwin/release/effectcraft.
set -euo pipefail
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$ROOT/packaging/macos"

DEST="$HOME/Applications"
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h | --help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

NAME="EffectCraft-Custom"
DISPLAY="EffectCraft Custom"
BUNDLE_ID="ai.storyteller.effectcraft.custom"
TARGET=aarch64-apple-darwin
export MACOSX_DEPLOYMENT_TARGET=11.0
export EFFECTCRAFT_DATA_NAME="$DISPLAY"
SHORT_VERSION="${VERSION%%-*}"
WORK="$CARGO_TARGET_DIR/custom-app"
APP="$WORK/$NAME.app"

echo "==> $NAME $VERSION ($TARGET), commit ${EFFECTCRAFT_BUILD_SHA:-unknown}"
if [ "$SKIP_BUILD" = 0 ]; then
  (cd "$ROOT" && cargo build --release --locked -p effectcraft --target "$TARGET")
fi
BIN="$CARGO_TARGET_DIR/$TARGET/release/effectcraft"
[ -x "$BIN" ] || { echo "error: no $BIN (build first)" >&2; exit 1; }

rm -rf "$WORK"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/$NAME"
cp "$ROOT/assets/app-icon/effectcraft.icns" "$APP/Contents/Resources/EffectCraft.icns"
PLIST="$APP/Contents/Info.plist"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@SHORT_VERSION@/$SHORT_VERSION/g" \
  -e "s/@BUILD_SHA@/${EFFECTCRAFT_BUILD_SHA:-unknown}/g" "$HERE/Info.plist.in" >"$PLIST"
plutil -replace CFBundleIdentifier -string "$BUNDLE_ID" "$PLIST"
plutil -replace CFBundleName -string "$DISPLAY" "$PLIST"
plutil -replace CFBundleDisplayName -string "$DISPLAY" "$PLIST"
plutil -replace CFBundleExecutable -string "$NAME" "$PLIST"
plutil -replace CFBundleDocumentTypes.0.LSHandlerRank -string Alternate "$PLIST"
# The installed release declares (exports) the project type; this build only imports it.
ENTRY="$(plutil -extract UTExportedTypeDeclarations.0 xml1 -o - "$PLIST" | sed -e '1,3d' -e '$d')"
N="$(plutil -extract UTImportedTypeDeclarations raw -o - "$PLIST" 2>/dev/null || echo 0)"
if [ "$N" = 0 ]; then plutil -insert UTImportedTypeDeclarations -array "$PLIST"; fi
plutil -insert "UTImportedTypeDeclarations.$N" -xml "$ENTRY" "$PLIST"
plutil -remove UTExportedTypeDeclarations "$PLIST"
plutil -lint "$PLIST"
printf 'APPL????' >"$APP/Contents/PkgInfo"

codesign --force --sign - --timestamp=none --options runtime --entitlements "$HERE/entitlements.plist" "$APP/Contents/MacOS/$NAME"
codesign --force --sign - --timestamp=none --options runtime --entitlements "$HERE/entitlements.plist" "$APP"
codesign --verify --strict --deep --verbose=2 "$APP"

mkdir -p "$DEST"
case "$DEST/$NAME.app" in
  */EffectCraft.app) echo "error: refusing to write over EffectCraft.app" >&2; exit 1 ;;
esac
rm -rf "${DEST:?}/$NAME.app"
ditto "$APP" "$DEST/$NAME.app"
rm -rf "$WORK"
echo "==> $DEST/$NAME.app"
