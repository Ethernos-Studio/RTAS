// 用 Spine 3.8 官方 JS 运行时把敌人的 .skel/.atlas 渲染成一张静态 PNG（setup pose）。
//
// 为什么是 3.8：ArkUnpacker 文档写明明日方舟小人是 Spine **3.8**；npm 上的
// @esotericsoftware/spine-canvas 是 4.x，读 3.8 的 skel 会报
// "String in string table must not be null"。
//
// 用法：
//   node render38.js <atlas> <skel> <out.png> [宽] [高]

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { createCanvas, loadImage } = require("@napi-rs/canvas");

function loadSpine() {
  const sandbox = {
    console, Math, Date, JSON, Object, Array, String, Number, Boolean, Error,
    Float32Array, Uint16Array, Uint32Array, Int32Array, Uint8Array, ArrayBuffer,
    performance: { now: () => Date.now() },
    window: {},
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext("var spine = {};", sandbox);
  for (const file of ["spine-core.js", "spine-canvas.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "vendor", file), "utf8"), sandbox, { filename: file });
  }
  return sandbox.spine;
}

const spine = loadSpine();
if (!spine || !spine.SkeletonBinary) {
  console.error("spine 运行时没加载成功");
  process.exit(3);
}

/** atlas 里的贴图页文件名：`<name>` 后紧跟 `size:` 和 `format:` 三行才是页头 */
function pageNames(atlasText) {
  const lines = atlasText.split(/\r\n|\r|\n/);
  const names = [];
  for (let i = 1; i + 1 < lines.length; i += 1) {
    const current = lines[i].trim();
    const next = lines[i + 1].trim();
    if (current.startsWith("size:") && next.startsWith("format:")) {
      names.push(lines[i - 1].trim());
    }
  }
  return names;
}

async function main() {
  const [atlasPath, skelPath, outPath, wArg, hArg] = process.argv.slice(2);
  if (!atlasPath || !skelPath || !outPath) {
    console.error("用法: node render38.js <atlas> <skel> <out.png> [宽] [高]");
    process.exit(2);
  }
  const width = parseInt(wArg || "512", 10);
  const height = parseInt(hArg || "512", 10);
  const dir = path.dirname(atlasPath);
  const atlasText = fs.readFileSync(atlasPath, "utf8");

  // 先同步把贴图画进 canvas（3.8 的 loader 是同步回调，取纹理时就要尺寸）
  const textures = {};
  for (const name of pageNames(atlasText)) {
    const file = path.join(dir, name);
    if (!fs.existsSync(file)) {
      console.warn("缺贴图:", name);
      continue;
    }
    const image = await loadImage(file);
    const canvas = createCanvas(image.width, image.height);
    canvas.getContext("2d").drawImage(image, 0, 0);
    // 传 canvas（不是 ctx）：3.8 里 page.width = texture.getImage().width，
    // 传 ctx 会拿到 undefined，导致所有 UV 错位、大部分部件画不出来
    textures[name] = new spine.canvas.CanvasTexture(canvas);
  }

  const atlas = new spine.TextureAtlas(atlasText, (name) => {
    const texture = textures[name];
    if (!texture) throw new Error(`贴图没加载: ${name}`);
    return texture;
  });

  const binary = new spine.SkeletonBinary(new spine.AtlasAttachmentLoader(atlas));
  const skeletonData = binary.readSkeletonData(new Uint8Array(fs.readFileSync(skelPath)));
  const skeleton = new spine.Skeleton(skeletonData);

  // 敌人的可见部件大多挂在动画里（setup pose 常常是空的）；另外**皮肤名不一定是 default**
  // （例：A3 术师的皮肤叫 Wizard，用默认皮肤会得到空骨架 -> bounds 变 -Infinity）。
  // 所以这里按"皮肤 × 动画"组合试一遍，取第一个能算出有效包围盒的。
  const skinNames = skeletonData.skins.map((s) => s.name);
  const animNames = skeletonData.animations.map((a) => a.name);
  const idleName =
    animNames.find((n) => /^idle$/i.test(n)) ||
    animNames.find((n) => /idle/i.test(n)) ||
    animNames.find((n) => /move|walk/i.test(n)) ||
    animNames[0];

  const offset = new spine.Vector2();
  const size = new spine.Vector2();
  let usedSkin = null;
  for (const skinName of skinNames.length ? skinNames : [null]) {
    if (skinName) skeleton.setSkinByName(skinName);
    skeleton.setToSetupPose();
    if (idleName) {
      const state = new spine.AnimationState(new spine.AnimationStateData(skeletonData));
      state.setAnimation(0, idleName, true);
      state.update(0);
      state.apply(skeleton);
    }
    skeleton.updateWorldTransform();
    skeleton.getBounds(offset, size, []);
    if (Number.isFinite(size.x) && Number.isFinite(size.y) && size.x > 1 && size.y > 1) {
      usedSkin = skinName || "(default)";
      break;
    }
  }
  if (!usedSkin) {
    console.error("所有皮肤都算不出有效包围盒（模型可能只在特定动画里才有几何）");
    process.exit(4);
  }
  const margin = 0.92;
  const scale = Math.min((width * margin) / Math.max(size.x, 1), (height * margin) / Math.max(size.y, 1));
  const tx = (width - size.x * scale) / 2 - offset.x * scale;
  const ty = (height - size.y * scale) / 2 - offset.y * scale;

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const renderer = new spine.canvas.SkeletonRenderer(ctx);
  renderer.premultipliedAlpha = true;
  // 3.8 的 draw(skeleton) 不接受变换参数；注意 Y 轴：canvas 向下、spine 向上，要翻一下
  renderer.triangleRendering = true;
  ctx.save();
  ctx.translate(width / 2, height / 2);
  ctx.scale(scale, -scale);
  ctx.translate(-(offset.x + size.x / 2), -(offset.y + size.y / 2));
  renderer.draw(skeleton);
  ctx.restore();

  fs.writeFileSync(outPath, canvas.toBuffer("image/png"));
  console.log(
    `ok ${path.basename(outPath)} ${width}x${height} bounds=${size.x.toFixed(0)}x${size.y.toFixed(0)} scale=${scale.toFixed(2)}`
  );
}

main().catch((err) => {
  console.error("渲染失败:", err && err.message ? err.message : err);
  process.exit(1);
});
