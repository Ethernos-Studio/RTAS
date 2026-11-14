// 用官方 spine 运行时把敌人的 .skel/.atlas 渲染成一张静态 PNG（setup pose）。
//
// 用法：
//   node render.js <atlas> <skel> <out.png> [宽] [高]
//
// 为什么这么做：游戏里敌人立绘是 Spine（.skel + .atlas + 贴图），
// 打包出来的 PNG 是 atlas 整页碎片，不能直接当参考图；渲染一帧才有干净全身图。

const fs = require("fs");
const path = require("path");
const { createCanvas, loadImage } = require("@napi-rs/canvas");
const spine = require("@esotericsoftware/spine-canvas");

async function main() {
  const [atlasPath, skelPath, outPath, wArg, hArg] = process.argv.slice(2);
  if (!atlasPath || !skelPath || !outPath) {
    console.error("用法: node render.js <atlas> <skel> <out.png> [宽] [高]");
    process.exit(2);
  }
  const width = parseInt(wArg || "512", 10);
  const height = parseInt(hArg || "512", 10);
  const dir = path.dirname(atlasPath);

  // atlas 里引用的贴图页：按文件名从同目录读取
  const atlasText = fs.readFileSync(atlasPath, "utf8");
  const pages = {};
  const imageCache = new Map();
  const atlas = new spine.TextureAtlas(atlasText, (pageName) => {
    const file = path.join(dir, pageName);
    const canvas = createCanvas(1, 1);
    const ctx = canvas.getContext("2d");
    const texture = new spine.canvas.CanvasTexture(ctx);
    pages[pageName] = { canvas, ctx, texture };
    imageCache.set(pageName, file);
    // 先返回空纹理，等下面异步加载图片后重画
    return texture;
  });

  // 把贴图真的画进纹理
  for (const [pageName, record] of Object.entries(pages)) {
    const file = imageCache.get(pageName);
    if (!fs.existsSync(file)) continue;
    const image = await loadImage(file);
    record.canvas.width = image.width;
    record.canvas.height = image.height;
    record.ctx.drawImage(image, 0, 0);
    record.texture.setFilters(spine.TextureFilter.Linear, spine.TextureFilter.Linear);
  }

  const binary = new spine.SkeletonBinary(new spine.AtlasAttachmentLoader(atlas));
  const skelData = new Uint8Array(fs.readFileSync(skelPath));
  const skeletonData = binary.readSkeletonData(skelData);
  const skeleton = new spine.Skeleton(skeletonData);

  // setup pose 就是站立姿势；有些敌人默认是攻击姿态，可以换成 idle 动画第 0 帧
  if (skeletonData.animations.length > 0) {
    const idle = skeletonData.animations.find((a) => /idle/i.test(a.name));
    if (idle) {
      skeleton.setToSetupPose();
      const state = new spine.AnimationState(new spine.AnimationStateData(skeletonData));
      state.setAnimation(0, idle, true);
      state.update(0);
      state.apply(skeleton);
    }
  }
  skeleton.updateWorldTransform(spine.Physics.none ? spine.Physics.none : undefined);

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  const renderer = new spine.canvas.SkeletonRenderer(ctx);
  renderer.premultipliedAlpha = true;
  renderer.triangleRendering = true;

  // 按包围盒自动居中缩放
  skeleton.setToSetupPose();
  skeleton.updateWorldTransform();
  const offset = new spine.Vector2();
  const size = new spine.Vector2();
  skeleton.getBounds(offset, size, []);
  const margin = 0.9;
  const scale = Math.min((width * margin) / Math.max(size.x, 1), (height * margin) / Math.max(size.y, 1));
  const tx = (width - size.x * scale) / 2 - offset.x * scale;
  const ty = (height - size.y * scale) / 2 - offset.y * scale;

  renderer.draw(skeleton, tx, ty, scale);
  fs.writeFileSync(outPath, canvas.toBuffer("image/png"));
  console.log(`ok ${path.basename(outPath)} ${width}x${height} scale=${scale.toFixed(2)}`);
}

main().catch((err) => {
  console.error("渲染失败:", err && err.message ? err.message : err);
  process.exit(1);
});
